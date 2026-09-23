"""Measurements are CUDA-only. Both wall and CUDA-event intervals are recorded.

CUDA event intervals are not a claim about pure kernel execution: they can include
queue idle gaps. Library/workspace allocations not owned by PyTorch are not fully
represented by max_memory_allocated. Device usage is sampled before/after only.
"""
import gc
import time
import torch
from .ops import run_plan,validate_inputs
from .common import MIB


def device_used():
    try:
        import pynvml
        pynvml.nvmlInit()
        return int(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(0)).used)
    except Exception:return None


def make_inputs(shape,seed,device='cuda'):
    g=torch.Generator(device=device).manual_seed(seed)
    n,v,d=(shape[k] for k in ('n','v','d'))
    h=torch.randn(n,d,device=device,generator=g,dtype=torch.float32).requires_grad_()
    w=(torch.randn(v,d,device=device,generator=g,dtype=torch.float32)/d**.5).requires_grad_()
    y=torch.randint(v,(n,),device=device,generator=g)
    validate_inputs(h,w,y)
    return h,w,y


def time_block(step,clear,validate,warmup,iterations):
    if iterations<1 or warmup<0:raise ValueError('Invalid measurement lengths.')
    for _ in range(warmup):step()
    torch.cuda.synchronize()
    clear();gc.collect();torch.cuda.synchronize()
    baseline=int(torch.cuda.memory_allocated())
    reserved_before=int(torch.cuda.memory_reserved())
    device_before=device_used()
    torch.cuda.reset_peak_memory_stats()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    t0=time.perf_counter();start.record()
    for _ in range(iterations):step()
    end.record();torch.cuda.synchronize();wall=time.perf_counter()-t0
    peak=int(torch.cuda.max_memory_allocated());respeak=int(torch.cuda.max_memory_reserved())
    gpu_ms=float(start.elapsed_time(end))/iterations
    validate() # deliberately outside timing
    return {'wall_ms':wall*1000/iterations,'cuda_event_ms':gpu_ms,'block_wall_s':wall,
            'iterations':iterations,'warmup':warmup,'baseline_allocated_bytes':baseline,
            'peak_allocated_bytes':peak,'extra_peak_allocated_bytes':max(0,peak-baseline),
            'reserved_before_bytes':reserved_before,'peak_reserved_bytes':respeak,
            'device_used_before_bytes':device_before,'device_used_after_bytes':device_used()}


def operator_trial(shape,plan,seed,cfg):
    gc.collect();torch.cuda.empty_cache()
    h,w,y=make_inputs(shape,seed);state={}
    def clear():h.grad=None;w.grad=None;state.clear()
    def step():
        h.grad=None;w.grad=None
        loss=run_plan(plan,h,w,y,dtype=torch.float16)
        (loss*cfg['loss_scale']).backward()
        state['loss']=loss.detach()
    def validate():
        for name,t in [('loss',state['loss']),('hidden_gradient',h.grad),('weight_gradient',w.grad)]:
            if t is None or not bool(torch.isfinite(t).all().item()):
                raise RuntimeError('Nonfinite/missing '+name)
    result=time_block(step,clear,validate,cfg['warmup'],cfg['iterations'])
    result.update(loss=float(state['loss'].item()),tokens_per_s=shape['n']*1000/result['wall_ms'],
                  static_loss_scale=cfg['loss_scale'])
    clear();del h,w,y;gc.collect();torch.cuda.empty_cache()
    return result
