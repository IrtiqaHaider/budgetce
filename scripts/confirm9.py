#!/usr/bin/env python3
"""BudgetCE targeted follow-up: nine longer, matched-chunk training trials.
Uses the existing source and CUDA extension; does not alter prior results or kernels.
Run from the prepared project environment with --prior pointing at the original results.
"""
import argparse
import collections
import copy
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import sys
import traceback
import zipfile

METHODS = ('native', 'torch_1024', 'cuda_1024')


def build_schedule(seed):
    schedule=[]
    for repeat in range(3):
        methods=list(METHODS)
        random.Random(seed+repeat+9311).shuffle(methods)
        for method in methods:
            schedule.append(dict(sequence=1024,method=method,repeat=repeat,seed=seed+repeat*19))
    return schedule


class FixedCUDAPlan:
    def select(self, shape, budget_bytes):
        return {'plan':'cuda:1024','policy':'fixed_chunk_no_adaptive_claim',
                'operator_memory_budget_enforced':False}


def internal_method(method):
    if method=='native':return 'native'
    if method=='torch_1024':return 'torch_chunked'
    if method=='cuda_1024':return 'adaptive'  # existing engine dispatch; FixedCUDAPlan removes adaptation
    raise ValueError(f'Unknown measured method: {method}')


def summarize(output, schedule):
    output=Path(output);rows=[];coverage=[]
    for item in schedule:
        path=output/'raw'/f"r{item['repeat']}-{item['method']}.json"
        row=json.loads(path.read_text()) if path.exists() else None
        coverage.append(dict(**item,status=row['status'] if row else 'missing'))
        if row:rows.append(row)
    def table(name,rs):
        fields=list(dict.fromkeys(k for r in rs for k in r))
        if not fields:return
        with (output/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)
    table('coverage.csv',coverage)
    summary=[];pairs=[]
    for method in METHODS:
        good=[r for r in rows if r['method']==method and r['status']=='ok']
        if not good:continue
        tokens=sum(r['total_measured_tokens'] for r in good);seconds=sum(r['block_wall_s'] for r in good)
        times=[r['wall_ms'] for r in good]
        summary.append(dict(method=method,valid_repeats=len(good),eligible=len(good)==3,
            wall_ms=seconds*1000/sum(r['iterations'] for r in good),tokens_per_s=tokens/seconds,
            min_ms=min(times),max_ms=max(times),cv=statistics.stdev(times)/statistics.mean(times) if len(times)>1 else None,
            peak_allocated_mib=max(r['peak_allocated_bytes'] for r in good)/1024**2))
    for repeat in range(3):
        d={r['method']:r for r in rows if r['repeat']==repeat and r['status']=='ok'}
        if set(d)==set(METHODS):
            pairs.append(dict(repeat=repeat,cuda_vs_native=d['native']['wall_ms']/d['cuda_1024']['wall_ms'],
                              cuda_vs_torch_matched_chunk=d['torch_1024']['wall_ms']/d['cuda_1024']['wall_ms'],
                              cuda_loss_trajectory_max_abs=max(abs(a-b) for a,b in zip(d['cuda_1024']['losses'],d['native']['losses']))))
    table('summary.csv',summary);table('paired_repeat_comparisons.csv',pairs)
    print('\nSUMMARY (new measurements only; old and new windows are NOT pooled)')
    for row in summary:print(row,flush=True)
    return coverage,summary,pairs


def export(output):
    output=Path(output)
    checksums={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(output.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json' and not p.name.endswith('.tmp')}
    (output/'SHA256SUMS.json').write_text(json.dumps(checksums,indent=2,sort_keys=True)+'\n')
    path=output.parent/(output.name+'_results.zip')
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file() and not p.name.endswith('.tmp'):z.write(p,p.relative_to(output.parent))
    print('Export:',path,flush=True)
    return path


def preflight(output):
    """Actual follow-up operator shape and chunk, using the original tolerances."""
    import torch
    from budgetce.measure import make_inputs
    from budgetce.ops import run_plan
    from budgetce.common import write_json
    h,w,y=make_inputs(dict(n=1024,v=32768,d=256),987431)
    ref=run_plan('native',h,w,y,dtype=torch.float16)
    refg=torch.autograd.grad(128*ref,(h,w))
    checks=[]
    rel=lambda a,b:float((a.detach().double()-b.detach().double()).norm()/a.detach().double().norm().clamp_min(1e-12))
    for plan in ('torch:1024','cuda:1024'):
        loss=run_plan(plan,h,w,y,dtype=torch.float16)
        grads=torch.autograd.grad(128*loss,(h,w))
        errs=[rel(a,b) for a,b in zip(refg,grads)]
        loss_error=rel(ref,loss)
        check=dict(plan=plan,loss_relative_error=loss_error,gradient_relative_errors=errs,
                   loss_tolerance=5e-4,gradient_tolerance=5e-3,
                   passed=math.isfinite(loss_error) and loss_error<=5e-4 and all(math.isfinite(e) and e<=5e-3 for e in errs))
        checks.append(check)
    record=dict(status='passed' if all(c['passed'] for c in checks) else 'failed',checks=checks,
                purpose='diagnostic_not_training_measurement')
    write_json(Path(output)/'preflight.json',record)
    if record['status']!='passed':raise RuntimeError('Matched-chunk numerical gate failed; do not relax tolerances.')
    print('Matched-chunk GPU preflight: 2/2 passed.',flush=True)


def run(prior):
    import torch
    from budgetce.common import ROOT, source_hash, configure_torch, environment, write_json, digest, require_gate
    from budgetce.train import training_trial
    prior=Path(prior)
    old_cfg=json.loads((prior/'config.json').read_text())
    old_env=json.loads((prior/'environment.json').read_text())
    require_gate(prior)
    if source_hash()!=old_env['source_sha256']:
        raise RuntimeError('Prepared source differs from the audited run; no results were changed.')
    configure_torch(); env=environment()
    expected='858c1389d6b334900d7ed06a232e0b4d4a5b94993bb64fa66583638246b4f8e0'
    if source_hash()!=expected:raise RuntimeError('This add-on is for the audited BudgetCE source revision.')
    cfg=copy.deepcopy(old_cfg)
    cfg.update(profile='matched_chunk_training_followup',training_sequences=[1024],training_updates=100,
               training_warmup=20,training_repeats=3,torch_chunk=1024,run_profiler=False)
    schedule=build_schedule(cfg['seed'])
    helper_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    run_id=digest(dict(config=cfg,environment=env,helper_sha256=helper_hash))[:12]
    output=prior.parent/('budgetce_matched9_'+run_id)
    output.mkdir(parents=True,exist_ok=True);(output/'raw').mkdir(exist_ok=True)
    for name,value in [('config.json',cfg),('environment.json',env),('schedule.json',schedule)]:
        path=output/name
        if path.exists() and json.loads(path.read_text())!=value:raise RuntimeError('Run metadata mismatch: '+name)
        write_json(path,value)
    shutil.copyfile(__file__,output/'confirm9.py')
    for p in sorted((ROOT/'budgetce').rglob('*')):
        if p.is_file() and p.suffix in ('.py','.cu'):
            dest=output/'source'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    write_json(output/'PROTOCOL.json',dict(
        prior_results=str(prior),prior_source_sha256=old_env['source_sha256'],helper_sha256=helper_hash,
        goal='Confirm S=1024 whole-training effect with native, torch:1024, cuda:1024 and longer blocks',
        changes=['100 timed updates; 20 warmup','both chunked paths use chunk=1024','CUDA plan fixed; no adaptive attribution'],
        selected_after_inspecting_original_results=True,
        same_prior_environment=env==old_env,
        limitations=['one shape, GPU and session','fixed CUDA candidate, not an optimized library comparison','do not pool different measurement windows','synthetic next-token objective; no quality/convergence claim']))
    print('Follow-up: 9 trials, S=1024, 20 warmup + 100 measured updates each.',flush=True)
    print('Prior checkpoints remain unchanged:',prior,flush=True)
    try:
        preflight(output)
        for index,item in enumerate(schedule,1):
            path=output/'raw'/f"r{item['repeat']}-{item['method']}.json"
            if path.exists():
                saved=json.loads(path.read_text())
                if saved['status']!='ok':raise RuntimeError('Prior failed checkpoint retained; no automatic retry: '+str(path))
                print('Skip existing valid checkpoint:',path.name,flush=True);continue
            record=dict(item,status='pending')
            try:
                result=training_trial(item['sequence'],internal_method(item['method']),item['seed'],cfg,FixedCUDAPlan())
                record.update(result,status='ok')
                if not record['within_training_budget']:
                    record['status']='budget_exceeded';raise RuntimeError('Whole-training allocated-memory budget exceeded.')
            except BaseException as e:
                record.update(status='error' if record['status']=='pending' else record['status'],error=repr(e),traceback=traceback.format_exc())
                write_json(path,record);raise
            write_json(path,record)
            print(f"{index}/9 {item['method']} r={item['repeat']+1}: {record['tokens_per_s']:.1f} tokens/s; "
                  f"{record['peak_allocated_bytes']/1024**2:.1f} MiB allocated; {record['optimizer_steps']}/{record['expected_optimizer_steps']} optimizer updates",flush=True)
    finally:
        summarize(output,schedule); export(output)
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prior',required=True);a=p.parse_args();run(a.prior)
