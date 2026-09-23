"""Bounded experiments. Existing successes AND failures are immutable checkpoints."""
import gc
import itertools
import random
import time
import traceback
from pathlib import Path
import torch
from .common import read_json,write_json,digest,plans,require_gate,MIB
from .measure import operator_trial
from .planner import Planner
from .train import training_trial


def checkpoint(out,phase,meta,fn):
    path=Path(out)/'raw'/phase/(digest(meta)[:20]+'.json')
    if path.exists():
        row=read_json(path)
        if row['status']=='error':raise RuntimeError(f'Prior failed checkpoint preserved: {path}. Inspect it; use a new run after a fix.')
        return row
    t0=time.perf_counter()
    row=dict(meta,status='pending')
    try:
        row.update(fn());row['status']='ok'
        if row.get('within_training_budget') is False:row['status']='budget_exceeded'
    except torch.cuda.OutOfMemoryError as exc:
        row.update(status='oom',error=str(exc))
        gc.collect();torch.cuda.empty_cache()
    except Exception as exc:
        row.update(status='error',error=repr(exc),traceback=traceback.format_exc())
        row['job_total_ms']=(time.perf_counter()-t0)*1000
        write_json(path,row)
        raise
    row['job_total_ms']=(time.perf_counter()-t0)*1000
    if phase=='calibration':row['calibration_total_ms']=row['job_total_ms']
    write_json(path,row)
    return row


def run_operators(out,cfg):
    require_gate(out)
    out=Path(out);p=plans(cfg);cal=[]
    jobs=list(itertools.product(cfg['calibration_shapes'],p))
    random.Random(cfg['seed']).shuffle(jobs)
    for k,(shape,plan) in enumerate(jobs,1):
        seed=cfg['seed']+int(digest(shape)[:6],16)
        meta=dict(phase='calibration',shape=shape,plan=plan,repeat=0,seed=seed)
        row=checkpoint(out,'calibration',meta,lambda:operator_trial(shape,plan,seed,cfg))
        cal.append(row)
        print(f"Calibration {k}/{len(jobs)} | {shape} {plan}: {row['status']} {row.get('wall_ms',0):.2f} ms",flush=True)
    t0=time.perf_counter()
    planner=Planner.fit(cal,cfg['planner_memory_margin'])
    # Planner needs custom-kernel calibration, not native/PyTorch baseline timings.
    planner.calibration_cost_ms=sum(r.get('calibration_total_ms',0) for r in cal if r['plan'].startswith('cuda:'))+(time.perf_counter()-t0)*1000
    model_path=out/'planner.json'
    if model_path.exists():
        # Checkpoints are fixed; avoid changing the plan due only to a fresh CPU fit timer.
        previous=Planner.from_dict(read_json(model_path))
        if previous.models!=planner.models:raise RuntimeError('Frozen planner differs; use a new result directory.')
        planner=previous
    else:write_json(model_path,planner.as_dict())
    decisions=[]
    for shape in cfg['evaluation_shapes']:
        for budget in cfg['budgets_mib']:
            t0=time.perf_counter();choice=planner.select(shape,int(budget*MIB))
            decisions.append(dict(shape=shape,budget_mib=budget,selection=choice,
                                  selection_overhead_ms=(time.perf_counter()-t0)*1000,
                                  predictions=planner.predictions(shape)))
    dp=out/'decisions_frozen_before_evaluation.json'
    if not dp.exists():write_json(dp,decisions)
    # Main plan never refits from these records.
    done=0;count=len(cfg['evaluation_shapes'])*len(p)*cfg['repeats']
    for repeat in range(cfg['repeats']):
        jobs=list(itertools.product(cfg['evaluation_shapes'],p));random.Random(cfg['seed']+repeat+191).shuffle(jobs)
        for shape,plan in jobs:
            seed=cfg['seed']+900000+repeat*101+int(digest(shape)[:6],16)
            meta=dict(phase='evaluation',shape=shape,plan=plan,repeat=repeat,seed=seed)
            row=checkpoint(out,'evaluation',meta,lambda:operator_trial(shape,plan,seed,cfg));done+=1
            print(f"Evaluation {done}/{count} | N={shape['n']} V={shape['v']} {plan} r={repeat+1}: "
                  f"{row['status']} {row.get('wall_ms',0):.2f} ms",flush=True)
    return planner


def run_training(out,cfg):
    require_gate(out);out=Path(out)
    if not (out/'planner.json').exists():raise RuntimeError('Run operator calibration before training.')
    planner=Planner.from_dict(read_json(out/'planner.json'))
    methods=['native','torch_chunked','adaptive'];count=len(methods)*len(cfg['training_sequences'])*cfg['training_repeats'];i=0
    for repeat in range(cfg['training_repeats']):
        jobs=list(itertools.product(cfg['training_sequences'],methods));random.Random(cfg['seed']+repeat+891).shuffle(jobs)
        for seq,method in jobs:
            seed=cfg['seed']+repeat*19
            meta=dict(phase='training',sequence=seq,method=method,repeat=repeat,seed=seed)
            row=checkpoint(out,'training',meta,lambda:training_trial(seq,method,seed,cfg,planner));i+=1
            print(f"Training {i}/{count} | S={seq} {method} r={repeat+1}: {row['status']} "
                  f"{row.get('tokens_per_s',0):.1f} tokens/s",flush=True)
