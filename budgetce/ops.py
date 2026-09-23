"""First-order linear CE. Dense objective; all-ignore mean deliberately returns zero.

Manual backward recomputes logits chunk-by-chunk, uses FP32 weight-gradient
accumulation (FP64 for CPU gradchecks) and never filters vocabulary contributions.
FP16 GEMM outputs/gradients are rounded, so equivalence is numerical, not bitwise.
"""
from contextlib import nullcontext
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from .extension import load_extension


def no_autocast(device):
    return torch.autocast(device_type=device.type, enabled=False) if device.type in ('cuda','cpu') else nullcontext()


def validate_inputs(h,w,y,ignore=-100,check_values=True):
    if h.ndim!=2 or w.ndim!=2 or y.ndim!=1 or h.shape[0]!=y.numel() or h.shape[1]!=w.shape[1]:
        raise ValueError('Expected h[N,D], weight[V,D], labels[N].')
    if min(*h.shape,*w.shape)<1:
        raise ValueError('Empty dimensions are not supported.')
    if h.device!=w.device or y.device!=h.device:
        raise ValueError('All inputs must share a device.')
    if not h.is_floating_point() or not w.is_floating_point() or y.dtype!=torch.long:
        raise TypeError('Floating hidden/weights and int64 labels required.')
    if not all(t.is_contiguous() for t in (h,w,y)):
        raise ValueError('Contiguous inputs required; make layout copies explicit before timing.')
    if check_values and bool(((y!=ignore)&((y<0)|(y>=w.shape[0]))).any().item()):
        raise ValueError('Target index outside the vocabulary.')


def normalizer(y,ignore,reduction,dtype):
    if reduction=='sum':
        return torch.ones((),device=y.device,dtype=dtype)
    if reduction!='mean':
        raise ValueError('Only mean and sum are supported (no smoothing/class weights).')
    return (y!=ignore).sum().to(dtype).clamp_min(1)


def native_loss(h,w,y,*,dtype=None,ignore=-100,reduction='mean',validate=True):
    validate_inputs(h,w,y,ignore,validate)
    dtype=dtype or h.dtype
    with no_autocast(h.device):
        z=h.to(dtype) @ w.to(dtype).T
        acc=torch.float64 if dtype==torch.float64 else torch.float32
        return F.cross_entropy(z.to(acc),y,ignore_index=ignore,reduction='sum')/normalizer(y,ignore,reduction,acc)


class _Chunked(torch.autograd.Function):
    @staticmethod
    def forward(ctx,h,w,y,chunk,backend,dtype,ignore,reduction):
        acc=torch.float64 if dtype==torch.float64 else torch.float32
        with no_autocast(h.device):
            hc,wc=h.to(dtype),w.to(dtype)
            den=normalizer(y,ignore,reduction,acc)
            total=torch.zeros((),device=h.device,dtype=acc)
            ext=load_extension() if backend=='cuda' else None
            for start in range(0,h.shape[0],chunk):
                z=hc[start:start+chunk] @ wc.T
                yc=y[start:start+chunk]
                if ext is not None:
                    total.add_(ext.row_loss(z,yc,ignore).sum())
                else:
                    total.add_(F.cross_entropy(z.to(acc),yc,ignore_index=ignore,reduction='sum'))
                del z
            ctx.save_for_backward(hc,wc,y,den)
            ctx.meta=(chunk,backend,ignore,h.dtype,w.dtype,acc)
            return total/den

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        hc,wc,y,den=ctx.saved_tensors
        chunk,backend,ignore,hdtype,wdtype,acc=ctx.meta
        with no_autocast(hc.device):
            scale=(g.to(acc)/den).contiguous()
            dh=torch.empty_like(hc) if ctx.needs_input_grad[0] else None
            dw=torch.zeros(wc.shape,device=wc.device,dtype=acc) if ctx.needs_input_grad[1] else None
            ext=load_extension() if backend=='cuda' else None
            for start in range(0,hc.shape[0],chunk):
                hh=hc[start:start+chunk]; yy=y[start:start+chunk]
                z=hh @ wc.T
                if ext is not None:
                    ext.backward_inplace(z,yy,scale.to(torch.float32),ignore)
                    dz=z
                    del z
                else:
                    probs=torch.softmax(z.to(acc),dim=1)
                    good=yy!=ignore
                    safe=yy.masked_fill(~good,0)
                    probs.scatter_add_(1,safe[:,None],-torch.ones((len(yy),1),device=z.device,dtype=acc))
                    probs.mul_(good[:,None]).mul_(scale)
                    dz=probs.to(hc.dtype)
                    del probs,z
                if dh is not None:
                    torch.mm(dz,wc,out=dh[start:start+chunk])
                if dw is not None:
                    # FP16 GEMM temporary, FP32 accumulation; no per-chunk FP32 copy of the full dW.
                    partial=dz.T @ hh
                    dw.add_(partial)
                    del partial
                del dz
        return (dh.to(hdtype) if dh is not None else None,
                dw.to(wdtype) if dw is not None else None,
                None,None,None,None,None,None)


def chunked_loss(h,w,y,chunk=256,backend='torch',*,dtype=None,ignore=-100,reduction='mean',validate=True):
    validate_inputs(h,w,y,ignore,validate)
    if not isinstance(chunk,int) or isinstance(chunk,bool) or chunk<1:
        raise ValueError('chunk must be a positive integer.')
    if backend not in ('torch','cuda'):
        raise ValueError('backend must be torch or cuda.')
    dtype=dtype or h.dtype
    if backend=='cuda' and (h.device.type!='cuda' or dtype not in (torch.float16,torch.float32)):
        raise ValueError('Custom CUDA supports FP16/FP32 compute on CUDA only.')
    if reduction not in ('mean','sum'):
        raise ValueError('Only mean/sum are supported.')
    return _Chunked.apply(h,w,y,chunk,backend,dtype,ignore,reduction)


def run_plan(plan,h,w,y,*,dtype=None,validate=False):
    if plan=='native':
        return native_loss(h,w,y,dtype=dtype,validate=validate)
    backend,c=plan.split(':')
    return chunked_loss(h,w,y,int(c),backend,dtype=dtype,validate=validate)
