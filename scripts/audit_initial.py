#!/usr/bin/env python3
"""Offline audit of a BudgetCE result ZIP. Does not execute archived Python/CUDA.
Usage: python audit_budgetce.py results.zip output_directory
Requires Python >=3.10; numpy is used only for the cost-model rank diagnostic.
"""
import argparse
import collections
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path, PurePosixPath
import statistics
from zipfile import ZipFile

MIB = 1024 ** 2


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def close(a, b):
    return math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-7)


def write_csv(path, rows):
    if not rows:
        path.write_text('')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def load_archive(archive):
    with ZipFile(archive) as z:
        items = z.infolist()
        if len({i.filename for i in items}) != len(items):
            raise ValueError('Duplicate archive paths')
        if sum(i.file_size for i in items) > 128 * 1024**2:
            raise ValueError('Archive exceeds this audit size limit')
        for i in items:
            p = PurePosixPath(i.filename)
            if p.is_absolute() or '..' in p.parts or ((i.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError('Unsafe archive path')
        cfg_paths = [i.filename for i in items if i.filename.count('/') == 1 and i.filename.endswith('/config.json')]
        if len(cfg_paths) != 1:
            raise ValueError('Expected one results root')
        prefix = cfg_paths[0].rsplit('/', 1)[0] + '/'
        if any(not i.filename.startswith(prefix) for i in items):
            raise ValueError('Unexpected file outside root')
        return {i.filename[len(prefix):]: z.read(i) for i in items if not i.is_dir()}


def summarize(rows):
    durations = [r['wall_ms'] for r in rows]
    avg = sum(r['block_wall_s'] for r in rows) * 1000 / sum(r['iterations'] for r in rows)
    return dict(
        trials=len(rows), wall_ms=avg, min_trial_ms=min(durations), max_trial_ms=max(durations),
        trial_cv=statistics.stdev(durations) / statistics.mean(durations) if len(durations)>1 else None,
        peak_allocated_mib=max(r['peak_allocated_bytes'] for r in rows)/MIB,
        extra_peak_allocated_mib=max(r['extra_peak_allocated_bytes'] for r in rows)/MIB,
        peak_reserved_mib=max(r['peak_reserved_bytes'] for r in rows)/MIB,
        timed_block_seconds=sum(r['block_wall_s'] for r in rows),
    )


def audit(archive, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    files = load_archive(archive)
    obj = lambda name: json.loads(files[name])
    csv_rows = lambda name: list(csv.DictReader(files[name].decode().splitlines()))
    checks = []
    def check(name, condition, detail=''):
        checks.append(dict(check=name, passed=bool(condition), detail=str(detail)))
    cfg, env, gate, planner = (obj(p) for p in ('config.json','environment.json','validation/gpu.json','planner.json'))
    manifest = obj('SHA256SUMS.json')
    check('manifest_coverage', set(manifest) == set(files)-{'SHA256SUMS.json'}, len(manifest))
    bad = [k for k,h in manifest.items() if k not in files or hashlib.sha256(files[k]).hexdigest()!=h]
    check('all_manifest_hashes', not bad, bad)
    sources = {k[len('source/'):]:hashlib.sha256(v).hexdigest() for k,v in files.items()
               if k.startswith('source/budgetce/') and k.endswith(('.py','.cu'))}
    source_sha = digest(sources)
    check('source_hash_vs_environment_and_gate', source_sha == env['source_sha256'] == gate['source_sha256'], source_sha)
    check('exported_config_matches_source_confirm', cfg==obj('source/configs/confirm.json'))
    check('recorded_cpu_tests_passed', '38 passed' in files['validation/cpu-tests.txt'].decode())
    numerical = gate['checks']
    check('recorded_gpu_gate_passed', gate['status']=='passed' and len(numerical)==21 and all(c['passed'] for c in numerical),len(numerical))
    check('numeric_tolerances_rechecked', all(
        math.isfinite(c['loss_relative_error']) and c['loss_relative_error'] <= c['loss_tolerance'] and
        all(math.isfinite(e) and e<=c['gradient_tolerance'] for e in c['gradient_relative_errors'])
        for c in numerical if 'loss_relative_error' in c))
    rows = {ph:[obj(k) for k in sorted(files) if k.startswith('raw/'+ph+'/') and k.endswith('.json')]
            for ph in ('calibration','evaluation','training')}
    plans=['native',f"torch:{cfg['torch_chunk']}"]+[f'cuda:{c}' for c in cfg['chunks']]
    shape_key=lambda s:(s['n'],s['v'],s['d'])
    all_rows=[]
    for phase,rs in rows.items():
        if phase=='training':
            expected = {(seq,m,r) for seq,m,r in itertools.product(cfg['training_sequences'],['native','torch_chunked','adaptive'],range(cfg['training_repeats']))}
            keys=[(x['sequence'],x['method'],x['repeat']) for x in rs]
        else:
            shapes=cfg[('calibration' if phase=='calibration' else 'evaluation')+'_shapes']
            reps=range(1 if phase=='calibration' else cfg['repeats'])
            expected={(shape_key(s),p,r) for s,p,r in itertools.product(shapes,plans,reps)}
            keys=[(shape_key(x['shape']),x['plan'],x['repeat']) for x in rs]
        check(phase+'_coverage_and_uniqueness',set(keys)==expected and len(keys)==len(expected),f'{len(rs)}/{len(expected)}')
        check(phase+'_all_successful',all(x['status']=='ok' for x in rs))
        paired=collections.defaultdict(set)
        for x in rs:
            paired[(x['sequence'] if phase=='training' else shape_key(x['shape']),x['repeat'])].add(x['seed'])
        check(phase+'_paired_seed_metadata',all(len(seeds)==1 for seeds in paired.values()))
        all_rows.extend(rs)
    check('wall_time_formulas',all(close(r['wall_ms'],r['block_wall_s']*1000/r['iterations']) for r in all_rows))
    check('positive_finite_timing',all(math.isfinite(r[k]) and r[k]>0 for r in all_rows for k in ('wall_ms','cuda_event_ms','block_wall_s')))
    check('memory_arithmetic_and_order',all(
        r['peak_allocated_bytes']>=r['baseline_allocated_bytes']>=0 and
        r['peak_reserved_bytes']>=r['peak_allocated_bytes'] and
        r['extra_peak_allocated_bytes']==r['peak_allocated_bytes']-r['baseline_allocated_bytes'] for r in all_rows))
    check('operator_iterations_and_scaling',all(r['iterations']==cfg['iterations'] and r['warmup']==cfg['warmup'] and r['static_loss_scale']==cfg['loss_scale'] and math.isfinite(r['loss'])
          for phase in ('calibration','evaluation') for r in rows[phase]))
    check('operator_throughput_formulas',all(close(r['tokens_per_s'],r['shape']['n']*1000/r['wall_ms']) for phase in ('calibration','evaluation') for r in rows[phase]))
    train=rows['training']; total=cfg['training_warmup']+cfg['training_updates']
    check('training_optimizer_steps_no_skips',all(r['optimizer_steps']==r['expected_optimizer_steps']==total for r in train))
    check('training_measurement_counts',all(r['iterations']==cfg['training_updates'] and r['warmup']==cfg['training_warmup'] and len(r['losses'])==total for r in train))
    check('training_losses_and_scales_finite',all(all(math.isfinite(x) for x in r['losses']) and r['final_grad_scale']==cfg['loss_scale'] for r in train))
    check('training_token_throughput_formulas',all(r['total_measured_tokens']==cfg['training_updates']*cfg['training_batch']*cfg['training_accumulation']*r['sequence'] and close(r['tokens_per_s'],r['total_measured_tokens']/r['block_wall_s']) for r in train))
    check('training_memory_budget',all(r['within_training_budget'] and r['peak_allocated_bytes']<=r['training_budget_bytes']==int(env['gpu_total_bytes']*cfg['training_budget_fraction']) for r in train))
    check('training_shapes_parameter_counts_and_plan', len({r['parameter_count'] for r in train})==1 and all(r['training_shape']==dict(n=cfg['training_batch']*r['sequence'],v=cfg['training_vocab'],d=cfg['training_d']) and r['effective_batch_sequences']==cfg['training_batch']*cfg['training_accumulation'] and (r['method']!='adaptive' or r['plan']==r['selection']['plan']) for r in train))
    op=[]; tr=[]
    for s,p in itertools.product(cfg['evaluation_shapes'],plans):
        rr=[r for r in rows['evaluation'] if r['shape']==s and r['plan']==p]
        d=dict(**s,plan=p,**summarize(rr)); d['tokens_per_s']=s['n']*1000/d['wall_ms']; op.append(d)
    for seq,m in itertools.product(cfg['training_sequences'],['native','torch_chunked','adaptive']):
        rr=[r for r in train if r['sequence']==seq and r['method']==m]
        d=dict(sequence=seq,method=m,plan=rr[0]['plan'],**summarize(rr))
        d['tokens_per_s']=sum(r['total_measured_tokens'] for r in rr)/sum(r['block_wall_s'] for r in rr)
        tr.append(d)
    metrics=['wall_ms','tokens_per_s','min_trial_ms','max_trial_ms','trial_cv','peak_allocated_mib','extra_peak_allocated_mib','peak_reserved_mib']
    for name,new in [('operator',op),('training',tr)]:
        saved=csv_rows('analysis/'+name+'_summary.csv')
        keycols=['n','v','d','plan'] if name=='operator' else ['sequence','method']
        key=lambda r:tuple(str(r[k]) for k in keycols)
        olds={key(x):x for x in saved}
        check(name+'_summary_recomputed',len(olds)==len(new) and all(key(x) in olds and all(close(x[k],olds[key(x)][k]) for k in metrics) for x in new))
    calibrated={shape_key(x) for x in cfg['calibration_shapes']}
    evaluated={shape_key(x) for x in cfg['evaluation_shapes']}
    check('calibration_evaluation_shapes_disjoint',not calibrated&evaluated)
    check('planner_shapes_calibration_only',all({shape_key(s) for s in model['calibration_shapes']}==calibrated for model in planner['models'].values()))
    costs=sum(r['calibration_total_ms'] for r in rows['calibration'] if r['plan'].startswith('cuda:'))
    check('calibration_cost_includes_cuda_jobs',planner['calibration_cost_ms']>=costs,f"total_ms={planner['calibration_cost_ms']}; cuda_jobs_ms={costs}")
    def predicted(s,p):
        n,v,d=s['n'],s['v'],s['d']; c=min(n,int(p.split(':')[1])); model=planner['models'][p]
        f=[1,n*v*d/1e9,math.ceil(n/c)]
        est=8*v*d+8*n*d+2*c*v+32*c+4096
        return dict(plan=p,predicted_ms=max(1e-9,sum(a*b for a,b in zip(f,model['coefficients']))),predicted_extra_bytes=math.ceil(est*model['memory_multiplier']))
    decisions=obj('decisions_frozen_before_evaluation.json'); policy=[]; prediction_ok=True
    for dec in decisions:
        s=dec['shape']; budget=dec['budget_mib']; selected=dec['selection']; candidates=dec['predictions']
        for p in candidates:
            predicted_p=predicted(s,p['plan'])
            prediction_ok &= close(p['predicted_ms'],predicted_p['predicted_ms']) and p['predicted_extra_bytes']==predicted_p['predicted_extra_bytes']
        feasible=[p for p in candidates if p['predicted_extra_bytes']<=budget*MIB]
        bestpred=min(feasible,key=lambda p:(p['predicted_ms'],p['plan'])) if feasible else None
        prediction_ok &= selected==bestpred
        largest=max(feasible,key=lambda p:int(p['plan'].split(':')[1])) if feasible else None
        obs={r['plan']:r for r in op if shape_key(r)==shape_key(s) and r['plan'].startswith('cuda:')}
        feasible_obs=[r for r in obs.values() if r['extra_peak_allocated_mib']<=budget]
        oracle=min(feasible_obs,key=lambda r:r['wall_ms']) if feasible_obs else None
        observed=obs[selected['plan']] if selected else None
        r=dict(**s,budget_mib=budget,selected=selected['plan'] if selected else '',largest_predicted_feasible=largest['plan'] if largest else '',best_observed=oracle['plan'] if oracle else '',
               selected_matches_largest=bool(selected and largest and selected['plan']==largest['plan']),selection_overhead_ms=dec['selection_overhead_ms'])
        if observed:
            r.update(actual_ms=observed['wall_ms'],actual_extra_mib=observed['extra_peak_allocated_mib'],budget_met=observed['extra_peak_allocated_mib']<=budget,predicted_ms=selected['predicted_ms'],predicted_extra_mib=selected['predicted_extra_bytes']/MIB,
                     regret_vs_best_observed=observed['wall_ms']/oracle['wall_ms']-1 if oracle else None,
                     speedup_vs_largest=obs[largest['plan']]['wall_ms']/observed['wall_ms'] if largest else None,
                     speedup_vs_fixed_cuda=obs[f"cuda:{cfg['torch_chunk']}"]['wall_ms']/observed['wall_ms'])
            delta=obs[f"cuda:{cfg['torch_chunk']}"]['wall_ms']-observed['wall_ms']
            r['calibration_break_even_uses']=math.ceil(planner['calibration_cost_ms']/delta) if delta>0 else None
        native=next(x for x in op if shape_key(x)==shape_key(s) and x['plan']=='native')
        r['native_extra_mib']=native['extra_peak_allocated_mib']
        r['native_within_budget']=native['extra_peak_allocated_mib']<=budget
        if observed:r['selected_vs_native_speedup']=native['wall_ms']/observed['wall_ms']
        policy.append(r)
    check('frozen_decision_coverage',len(decisions)==len(evaluated)*len(cfg['budgets_mib']) and {(shape_key(x['shape']),x['budget_mib']) for x in decisions}==set(itertools.product(evaluated,cfg['budgets_mib'])))
    check('frozen_predictions_and_choices_recomputed',prediction_ok)
    check('all_selected_budgets_met',all(x['budget_met'] for x in policy))
    savedpolicy={(int(x['n']),int(x['v']),int(x['d']),int(x['budget_mib'])):x for x in csv_rows('analysis/policy_evaluation.csv')}
    check('policy_summary_recomputed',all((lambda old: all(old[k]==str(r[k]) for k in ('selected','best_observed','largest_predicted_feasible','budget_met')) and all(close(old[k],r[k]) for k in ('actual_ms','actual_extra_mib','predicted_ms','predicted_extra_mib','regret_vs_best_observed','speedup_vs_largest','speedup_vs_fixed_cuda')))(savedpolicy[(*shape_key(r),r['budget_mib'])]) for r in policy))
    coverage=csv_rows('analysis/coverage.csv')
    check('coverage_table_matches_complete_trials',len(coverage)==len(rows['evaluation'])+len(train) and all(x['status']=='ok' for x in coverage))
    loss_diff=[]; comparisons=[]; pairtrain=[]; comptrain=[]
    for s in cfg['evaluation_shapes']:
        ds={r['plan']:r for r in op if shape_key(r)==shape_key(s)}
        a,t,c,x=(ds[p] for p in ('native','torch:256','cuda:256','cuda:1024'))
        comparisons.append(dict(**s,native_ms=a['wall_ms'],cuda1024_ms=x['wall_ms'],cuda1024_vs_native_speedup=a['wall_ms']/x['wall_ms'],native_extra_mib=a['extra_peak_allocated_mib'],cuda1024_extra_mib=x['extra_peak_allocated_mib'],extra_memory_reduction=1-x['extra_peak_allocated_mib']/a['extra_peak_allocated_mib'],total_allocated_reduction=1-x['peak_allocated_mib']/a['peak_allocated_mib'],cuda256_vs_torch256_speedup=t['wall_ms']/c['wall_ms'],cuda256_vs_native_speedup=a['wall_ms']/c['wall_ms']))
        for rep in range(cfg['repeats']):
            rr=[r for r in rows['evaluation'] if r['shape']==s and r['repeat']==rep]
            native=next(r for r in rr if r['plan']=='native')
            loss_diff.extend(abs(r['loss']-native['loss']) for r in rr)
    for seq in cfg['training_sequences']:
        ds={r['method']:r for r in tr if r['sequence']==seq};a,c,t=(ds[p] for p in ('native','adaptive','torch_chunked'))
        comptrain.append(dict(sequence=seq,native_tokens_per_s=a['tokens_per_s'],cuda_tokens_per_s=c['tokens_per_s'],cuda_vs_native_speedup=c['tokens_per_s']/a['tokens_per_s'],native_peak_mib=a['peak_allocated_mib'],cuda_peak_mib=c['peak_allocated_mib'],memory_reduction=1-c['peak_allocated_mib']/a['peak_allocated_mib'],cuda_vs_torch256_speedup=t['wall_ms']/c['wall_ms'],cuda_trial_cv=c['trial_cv']))
        for rep in range(cfg['training_repeats']):
            rr={r['method']:r for r in train if r['sequence']==seq and r['repeat']==rep}
            a,c,t=(rr[p] for p in ('native','adaptive','torch_chunked'))
            pairtrain.append(dict(sequence=seq,repeat=rep,native_ms=a['wall_ms'],torch256_ms=t['wall_ms'],cuda1024_ms=c['wall_ms'],cuda_vs_native_speedup=a['wall_ms']/c['wall_ms'],cuda_vs_torch256_speedup=t['wall_ms']/c['wall_ms'],max_loss_trajectory_abs_difference=max(abs(x-y) for x,y in zip(a['losses'],c['losses']))))
    rank_notes=[]
    try:
        import numpy as np
        for p,model in planner['models'].items():
            xx=[[1,s['n']*s['v']*s['d']/1e9,math.ceil(s['n']/min(s['n'],int(p.split(':')[1])))] for s in model['calibration_shapes']]
            rank_notes.append(dict(plan=p,columns=3,rank=int(np.linalg.matrix_rank(xx))))
    except ImportError:
        rank_notes=[dict(note='numpy not installed; optional rank diagnostic skipped')]
    ev=dict(archive_name=Path(archive).name,archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),source_sha256=source_sha,checks_passed=sum(x['passed'] for x in checks),checks_total=len(checks),all_checks_passed=all(x['passed'] for x in checks),counts={k:len(v) for k,v in rows.items()},measured_training_updates=sum(r['iterations'] for r in train),measured_training_tokens=sum(r['total_measured_tokens'] for r in train),source_gpu_tests=len(numerical),max_reported_gradient_relative_error=max(e for c in numerical for e in c.get('gradient_relative_errors',[])),operator_paired_loss_max_abs=max(loss_diff),training_parameter_count=train[0]['parameter_count'],timed_seconds={k:sum(r['block_wall_s'] for r in v) for k,v in rows.items()},timed_block_ranges={k:[min(r['block_wall_s'] for r in v),max(r['block_wall_s'] for r in v)] for k,v in rows.items()},selector_matches_largest=sum(x['selected_matches_largest'] for x in policy),selector_matches_hindsight=sum(x['selected']==x['best_observed'] for x in policy),selector_decisions=len(policy),calibration_cost_ms=planner['calibration_cost_ms'],cost_model_rank=rank_notes,analysis_scope='Offline verification of recorded evidence; no independent GPU execution. Metadata pairing is not a rehash of unavailable tensor values.')
    for fname,data in [('audit_checks.csv',checks),('operator_recomputed.csv',op),('training_recomputed.csv',tr),('operator_comparisons.csv',comparisons),('training_comparisons.csv',comptrain),('training_repeat_ratios.csv',pairtrain),('policy_recomputed.csv',policy)]:write_csv(output/fname,data)
    (output/'audit_evidence.json').write_text(json.dumps(ev,indent=2,allow_nan=False)+'\n')
    for rel in ['config.json','environment.json','validation/gpu.json','validation/cpu-tests.txt','planner.json','decisions_frozen_before_evaluation.json']:
        p=output/'evidence'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(files[rel])
    print(json.dumps(ev,indent=2))
    if not ev['all_checks_passed']:
        raise AssertionError('Audit failures: '+repr([x for x in checks if not x['passed']]))
    return ev

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive');p.add_argument('output');a=p.parse_args();audit(a.archive,a.output)
