// Original implementation. FP32 reductions; no gradient filtering or fast-math.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

constexpr int THREADS = 256;
__device__ __forceinline__ float warp_sum(float x) {
    for (int off=16; off>0; off/=2) x += __shfl_down_sync(0xffffffff, x, off);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
    for (int off=16; off>0; off/=2) x = fmaxf(x, __shfl_down_sync(0xffffffff, x, off));
    return x;
}
// All threads in each block participate. Separate barriers before shared reuse.
template<bool MAXIMUM>
__device__ __forceinline__ float block_reduce(float x, float* shared) {
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    x = MAXIMUM ? warp_max(x) : warp_sum(x);
    if (lane==0) shared[warp]=x;
    __syncthreads();
    float v = threadIdx.x < THREADS/32 ? shared[lane] : (MAXIMUM ? -INFINITY : 0.f);
    if (warp==0) {
        v = MAXIMUM ? warp_max(v) : warp_sum(v);
        if (lane==0) shared[0]=v;
    }
    __syncthreads();
    v=shared[0];
    __syncthreads();
    return v;
}

template<typename scalar_t, bool BACKWARD>
__global__ void row_kernel(scalar_t* z, const int64_t* labels,
                           float* losses, const float* scale,
                           int64_t vocab, int64_t ignore) {
    const int64_t row=blockIdx.x, y=labels[row], offset=row*vocab;
    if (y==ignore) {
        if (BACKWARD) {
            for (int64_t j=threadIdx.x;j<vocab;j+=THREADS) z[offset+j]=scalar_t(0.f);
        } else if (threadIdx.x==0) losses[row]=0.f;
        return;
    }
    // Defensive guard even if the Python validation was bypassed.
    if (y<0 || y>=vocab) {
        if (BACKWARD) {
            for (int64_t j=threadIdx.x;j<vocab;j+=THREADS) z[offset+j]=scalar_t(NAN);
        } else if (threadIdx.x==0) losses[row]=NAN;
        return;
    }
    __shared__ float buf[THREADS/32];
    float local_max=-INFINITY;
    for (int64_t j=threadIdx.x;j<vocab;j+=THREADS)
        local_max=fmaxf(local_max,float(z[offset+j]));
    float maxval=block_reduce<true>(local_max,buf);
    float local_sum=0.f;
    for (int64_t j=threadIdx.x;j<vocab;j+=THREADS)
        local_sum+=expf(float(z[offset+j])-maxval);
    float denom=block_reduce<false>(local_sum,buf);
    if (BACKWARD) {
        const float s=scale[0];
        for (int64_t j=threadIdx.x;j<vocab;j+=THREADS) {
            float p=expf(float(z[offset+j])-maxval)/denom;
            z[offset+j]=scalar_t((p-(j==y?1.f:0.f))*s);
        }
    } else if (threadIdx.x==0) {
        losses[row]=(maxval-float(z[offset+y]))+logf(denom);
    }
}

void check_inputs(const torch::Tensor& z, const torch::Tensor& y) {
    TORCH_CHECK(z.is_cuda() && y.is_cuda(), "CUDA inputs required");
    TORCH_CHECK(z.device()==y.device(), "inputs must share a device");
    TORCH_CHECK(z.dim()==2 && z.size(0)>0 && z.size(1)>0, "nonempty [rows, vocab] logits required");
    TORCH_CHECK(z.size(0)<=2147483647, "too many rows for launch grid");
    TORCH_CHECK(y.dim()==1 && y.size(0)==z.size(0), "label shape mismatch");
    TORCH_CHECK(y.scalar_type()==at::kLong, "labels must be int64");
    TORCH_CHECK(z.scalar_type()==at::kFloat || z.scalar_type()==at::kHalf, "logits must be FP16 or FP32");
    TORCH_CHECK(z.is_contiguous() && y.is_contiguous(), "contiguous tensors required");
}

torch::Tensor row_loss(torch::Tensor z, torch::Tensor y, int64_t ignore) {
    check_inputs(z,y);
    const c10::cuda::CUDAGuard guard(z.device());
    auto loss=torch::empty({z.size(0)},z.options().dtype(at::kFloat));
    auto stream=at::cuda::getCurrentCUDAStream(z.get_device());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(z.scalar_type(), "budgetce_forward", [&] {
        row_kernel<scalar_t,false><<<z.size(0),THREADS,0,stream>>>(
            z.data_ptr<scalar_t>(), y.data_ptr<int64_t>(), loss.data_ptr<float>(),
            nullptr, z.size(1), ignore);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return loss;
}

void backward_inplace(torch::Tensor z, torch::Tensor y, torch::Tensor scale, int64_t ignore) {
    check_inputs(z,y);
    TORCH_CHECK(scale.is_cuda() && scale.device()==z.device() && scale.scalar_type()==at::kFloat
                && scale.numel()==1 && scale.is_contiguous(), "one contiguous FP32 CUDA scale required");
    const c10::cuda::CUDAGuard guard(z.device());
    auto stream=at::cuda::getCurrentCUDAStream(z.get_device());
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(z.scalar_type(), "budgetce_backward", [&] {
        row_kernel<scalar_t,true><<<z.size(0),THREADS,0,stream>>>(
            z.data_ptr<scalar_t>(),y.data_ptr<int64_t>(),nullptr,scale.data_ptr<float>(),z.size(1),ignore);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("row_loss", &row_loss, "FP32 row CE losses (FP16/FP32 logits)");
    m.def("backward_inplace", &backward_inplace, "Replace disposable logits with scaled gradients");
}
