"""Optional, separately instrumented operator traces. Never merged into performance tables."""
from pathlib import Path
import torch
from .common import write_json,require_gate
from .measure import make_inputs
from .ops import run_plan

def traces(out,cfg):
    require_gate(out);dest=Path(out)/'profiles';dest.mkdir(exist_ok=True)
    shape=cfg['evaluation_shapes'][0]
    for plan in ['native',f"cuda:{cfg['torch_chunk']}"]:
        h,w,y=make_inputs(shape,cfg['seed'])
        def step():
            h.grad=None;w.grad=None
            (run_plan(plan,h,w,y,dtype=torch.float16)*cfg['loss_scale']).backward()
        for _ in range(2):step()
        torch.cuda.synchronize()
        try:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                        record_shapes=True,profile_memory=True) as p:
                step();torch.cuda.synchronize()
            path=dest/(plan.replace(':','_')+'.json');p.export_chrome_trace(str(path))
            write_json(dest/(plan.replace(':','_')+'_status.json'),dict(status='ok',scope='instrumented; excluded from timing tables'))
        except Exception as exc:
            write_json(dest/(plan.replace(':','_')+'_status.json'),dict(status='unavailable',error=repr(exc)))
        del h,w,y
        torch.cuda.empty_cache()
