"""GPU gate. Numeric failures are recorded and stop the benchmark, not relaxed automatically."""
import gc
import time
from pathlib import Path
import torch
from .ops import native_loss,chunked_loss
from .extension import load_extension
from .common import write_json,source_hash,configure_torch
from .model import TinyDecoder


def relative(a,b):
    a=a.detach().double();b=b.detach().double()
    return float((a-b).norm()/a.norm().clamp_min(1e-12))


def validate_gpu(out):
    configure_torch();t0=time.perf_counter();rows=[]
    path=Path(out)/'validation/gpu.json'
    if not torch.cuda.is_available():raise RuntimeError('A real CUDA device is required for this gate.')
    load_extension(verbose=True)
    def record(name,loss_err,grad_errors,loss_tol,grad_tol):
        ok=loss_err<=loss_tol and max(grad_errors,default=0)<=grad_tol
        rows.append(dict(name=name,loss_relative_error=loss_err,gradient_relative_errors=grad_errors,
                         loss_tolerance=loss_tol,gradient_tolerance=grad_tol,passed=ok))
        if not ok:raise AssertionError(f'GPU correctness failed: {rows[-1]}')
        print('PASS',name,flush=True)
    try:
        for dtype in (torch.float32,torch.float16):
            for n,v,d,c in [(17,257,31,7),(33,65536,64,16),(7,12753,63,3)]:
                torch.manual_seed(97+n)
                h=torch.randn(n,d,device='cuda',requires_grad=True)
                w=(torch.randn(v,d,device='cuda')/d**.5).requires_grad_()
                y=torch.randint(v,(n,),device='cuda');y[0]=-100
                scale=128. if dtype==torch.float16 else 2.3
                ref=native_loss(h,w,y,dtype=dtype);rg=torch.autograd.grad(ref*scale,(h,w))
                for backend in ('torch','cuda'):
                    value=chunked_loss(h,w,y,c,backend,dtype=dtype)
                    gs=torch.autograd.grad(value*scale,(h,w))
                    record(f'{backend}-{dtype}-N{n}-V{v}',relative(ref,value),[relative(a,b) for a,b in zip(rg,gs)],
                           5e-4 if dtype==torch.float16 else 2e-5,
                           5e-3 if dtype==torch.float16 else 3e-5)
                del h,w,y,ref,rg,value,gs
        # Exercise the actual candidate chunk sizes, including chunk > N and a tail.
        torch.manual_seed(812)
        h=torch.randn(259,63,device='cuda',requires_grad=True)
        w=(torch.randn(4097,63,device='cuda')/63**.5).requires_grad_()
        y=torch.randint(4097,(259,),device='cuda');y[11]=-100
        reference=native_loss(h,w,y,dtype=torch.float16)
        reference_grad=torch.autograd.grad(128*reference,(h,w))
        for c in (64,256,1024):
            result=chunked_loss(h,w,y,c,'cuda',dtype=torch.float16)
            grads=torch.autograd.grad(128*result,(h,w))
            record(f'candidate-chunk-{c}',relative(reference,result),
                   [relative(a,b) for a,b in zip(reference_grad,grads)],5e-4,5e-3)
        del h,w,y,reference,reference_grad,result,grads
        for reduction in ('mean','sum'):
            h=torch.randn(9,19,device='cuda',requires_grad=True)
            w=torch.randn(31,19,device='cuda',requires_grad=True)
            y=torch.full((9,),-100,device='cuda',dtype=torch.long)
            loss=chunked_loss(h,w,y,4,'cuda',dtype=torch.float16,reduction=reduction)
            gs=torch.autograd.grad(128*loss,(h,w))
            if loss.item()!=0 or any(torch.count_nonzero(g).item()!=0 for g in gs):
                raise AssertionError('All-ignore contract failed.')
            rows.append(dict(name=f'all-ignore-{reduction}',passed=True))
        # Current-stream correctness and extreme finite logits, without changing tolerances.
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            h=torch.randn(11,13,device='cuda',requires_grad=True)
            w=(torch.randn(257,13,device='cuda')*15).requires_grad_()
            y=torch.arange(11,device='cuda')
            ref=native_loss(h,w,y,dtype=torch.float32,reduction='sum')
            val=chunked_loss(h,w,y,4,'cuda',dtype=torch.float32,reduction='sum')
            rg=torch.autograd.grad(ref,(h,w));vg=torch.autograd.grad(val,(h,w))
        torch.cuda.current_stream().wait_stream(stream)
        record('nondefault-stream-extreme-sum',relative(ref,val),[relative(a,b) for a,b in zip(rg,vg)],2e-5,5e-5)
        # Full-model gradients under autocast, rather than just standalone operators.
        torch.manual_seed(12)
        model=TinyDecoder(vocab=257,d=32,layers=1,heads=4,max_seq=17).cuda()
        x=torch.randint(257,(2,18),device='cuda');labels=x[:,1:].contiguous().flatten()
        gradients={};losses={}
        for plan in ('native','torch','cuda'):
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                hidden=model(x[:,:-1].contiguous())
                loss=native_loss(hidden,model.classifier,labels,dtype=torch.float16) if plan=='native' else chunked_loss(hidden,model.classifier,labels,7,plan,dtype=torch.float16)
            (128*loss).backward()
            gradients[plan]=[p.grad.detach().clone() for p in model.parameters()]
            losses[plan]=loss.detach()
        for backend in ('torch','cuda'):
            record('full-model-'+backend,relative(losses['native'],losses[backend]),
                   [relative(a,b) for a,b in zip(gradients['native'],gradients[backend])],5e-4,.015)
        # Explicit accumulation equivalence in FP32 with no dropout.
        model.zero_grad(set_to_none=True)
        hidden=model(x[:,:-1].contiguous())
        loss=native_loss(hidden,model.classifier,labels,dtype=torch.float32)
        loss.backward();reference=[p.grad.detach().clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        for i in range(2):
            h=model(x[i:i+1,:-1].contiguous())
            (chunked_loss(h,model.classifier,labels.reshape(2,17)[i].contiguous(),7,'cuda',dtype=torch.float32)/2).backward()
        es=[relative(a,p.grad) for a,p in zip(reference,model.parameters())]
        record('model-accumulation',0.,es,0.,.001)
        write_json(path,dict(status='passed',checks=rows,source_sha256=source_hash(),elapsed_s=time.perf_counter()-t0,
                             note='FP32/FP16 numerical gate on this device; first-order gradients only.'))
    except Exception as exc:
        write_json(path,dict(status='failed',checks=rows,error=repr(exc),source_sha256=source_hash()))
        raise
    finally:
        gc.collect();torch.cuda.empty_cache()
    print(f'GPU correctness: {len(rows)} checks passed.',flush=True)
