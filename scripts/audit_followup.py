#!/usr/bin/env python3
"""Independently verify the matched-chunk records. Never executes archived code."""
import collections
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import statistics
from audit_initial import load_archive, digest, close, write_csv

METHODS = ('native', 'torch_1024', 'cuda_1024')
MIB = 1024**2


def audit(archive, initial_archive, output, repository=None):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    files, initial = load_archive(archive), load_archive(initial_archive)
    obj = lambda name: json.loads(files[name])
    csvrows = lambda name: list(csv.DictReader(io.StringIO(files[name].decode())))
    cfg, env, protocol = (obj(n) for n in ('config.json', 'environment.json', 'PROTOCOL.json'))
    checks = []
    def check(name, passed, detail=''):
        checks.append({'check': name, 'passed': bool(passed), 'detail': str(detail)})
    manifest = obj('SHA256SUMS.json')
    check('manifest_coverage', set(manifest) == set(files)-{'SHA256SUMS.json'}, len(manifest))
    check('manifest_hashes', all(k in files and hashlib.sha256(files[k]).hexdigest()==v for k,v in manifest.items()))
    check('followup_runner_hash', hashlib.sha256(files['confirm9.py']).hexdigest()==protocol['helper_sha256'])
    src = {k[7:]:hashlib.sha256(v).hexdigest() for k,v in files.items() if k.startswith('source/budgetce/') and k.endswith(('.py','.cu'))}
    check('source_hash', digest(src)==env['source_sha256']==protocol['prior_source_sha256'])
    check('source_matches_initial', all(k in initial and v==initial[k] for k,v in files.items() if k.startswith('source/budgetce/')))
    old_cfg, old_env = (json.loads(initial[n]) for n in ('config.json','environment.json'))
    expected_cfg = dict(old_cfg, profile='matched_chunk_training_followup', training_sequences=[1024], training_updates=100,
                        training_warmup=20, training_repeats=3, torch_chunk=1024, run_profiler=False)
    check('configuration_changes_only_as_declared', cfg==expected_cfg)
    check('saved_environments_match', env==old_env and protocol['same_prior_environment'] is True)
    check('posthoc_selection_disclosed', protocol['selected_after_inspecting_original_results'] is True)
    if repository is not None:
        root=Path(repository)
        check('repository_engine_matches_recorded_bytes', all((root/k).is_file() and hashlib.sha256((root/k).read_bytes()).hexdigest()==v for k,v in src.items()))
        check('repository_followup_runner_matches_recorded_bytes', (root/'scripts/confirm9.py').read_bytes()==files['confirm9.py'])
    gate=obj('preflight.json'); nums=gate['checks']
    check('preflight_coverage', gate['status']=='passed' and len(nums)==2 and {r['plan'] for r in nums}=={'torch:1024','cuda:1024'})
    check('preflight_tolerances_rechecked', all(
        r['passed'] and r['loss_tolerance']==5e-4 and r['gradient_tolerance']==5e-3 and
        math.isfinite(r['loss_relative_error']) and r['loss_relative_error']<=r['loss_tolerance'] and
        len(r['gradient_relative_errors'])==2 and all(math.isfinite(v) and v<=r['gradient_tolerance'] for v in r['gradient_relative_errors']) for r in nums))
    rows=[obj(k) for k in sorted(files) if k.startswith('raw/') and k.endswith('.json')]
    expected={(m,r) for m,r in itertools.product(METHODS,range(3))}
    key=lambda r:(r['method'],r['repeat'])
    check('trial_coverage_unique',len(rows)==9 and {key(r) for r in rows}==expected)
    check('all_statuses_ok',all(r['status']=='ok' for r in rows))
    schedule=obj('schedule.json')
    check('schedule_coverage_seeds',len(schedule)==9 and {key(r) for r in schedule}==expected and
          all(r['seed']==cfg['seed']+r['repeat']*19 and r['sequence']==1024 for r in schedule))
    check('paired_seeds',all(r['seed']==cfg['seed']+r['repeat']*19 for r in rows))
    check('measured_warmup_counts',all(r['iterations']==100 and r['warmup']==20 for r in rows))
    check('optimizer_steps_no_skips',all(r['optimizer_steps']==r['expected_optimizer_steps']==120 for r in rows))
    check('finite_loss_trajectories',all(len(r['losses'])==120 and all(math.isfinite(x) for x in r['losses']) and r['final_grad_scale']==128 for r in rows))
    check('workload_shapes',all(r['sequence']==1024 and r['parameter_count']==20189696 and r['effective_batch_sequences']==2
                               and r['training_shape']==dict(n=1024,d=256,v=32768) for r in rows))
    plans={'native':'native','torch_1024':'torch:1024','cuda_1024':'cuda:1024'}
    check('matched_chunks_fixed_plan',all(r['plan']==plans[r['method']] and (r['method']!='cuda_1024' or
          r['selection']['policy']=='fixed_chunk_no_adaptive_claim' and r['selection']['operator_memory_budget_enforced'] is False) for r in rows))
    check('time_and_token_formulas',all(r['block_wall_s']>0 and r['cuda_event_ms']>0 and close(r['wall_ms'],r['block_wall_s']*1000/100) and
          r['total_measured_tokens']==204800 and close(r['tokens_per_s'],204800/r['block_wall_s']) for r in rows))
    check('memory_accounting',all(r['peak_reserved_bytes']>=r['peak_allocated_bytes']>=r['baseline_allocated_bytes']>=0 and
          r['extra_peak_allocated_bytes']==r['peak_allocated_bytes']-r['baseline_allocated_bytes'] for r in rows))
    check('whole_training_budget',all(r['within_training_budget'] and r['peak_allocated_bytes']<=r['training_budget_bytes']==int(env['gpu_total_bytes']*cfg['training_budget_fraction']) for r in rows))
    summary=[]
    for m in METHODS:
        rr=[r for r in rows if r['method']==m]
        if len(rr)!=3: continue
        seconds=sum(r['block_wall_s'] for r in rr); tokens=sum(r['total_measured_tokens'] for r in rr)
        ms=[r['wall_ms'] for r in rr]; rates=[r['tokens_per_s'] for r in rr]
        summary.append(dict(method=m,valid_repeats=len(rr),wall_ms=seconds*1000/sum(r['iterations'] for r in rr),
            tokens_per_s=tokens/seconds,min_ms=min(ms),max_ms=max(ms),cv=statistics.stdev(ms)/statistics.mean(ms),
            min_tokens_per_s=min(rates),max_tokens_per_s=max(rates),
            peak_allocated_mib=max(r['peak_allocated_bytes'] for r in rr)/MIB,
            peak_reserved_mib=max(r['peak_reserved_bytes'] for r in rr)/MIB,
            measured_updates=sum(r['iterations'] for r in rr),measured_tokens=tokens,measured_wall_s=seconds))
    saved={r['method']:r for r in csvrows('summary.csv')}
    cols=('wall_ms','tokens_per_s','min_ms','max_ms','cv','peak_allocated_mib')
    check('summary_recomputed',len(saved)==len(summary)==3 and all(s['method'] in saved and
          saved[s['method']]['eligible']=='True' and saved[s['method']]['valid_repeats']=='3' and
          all(close(saved[s['method']][k],s[k]) for k in cols) for s in summary))
    pairs=[]
    for rep in range(3):
        d={r['method']:r for r in rows if r['repeat']==rep}
        if set(d)!=set(METHODS):continue
        a,t,c=(d[m] for m in METHODS)
        pairs.append(dict(repeat=rep,cuda_vs_native=a['wall_ms']/c['wall_ms'],cuda_vs_torch_matched_chunk=t['wall_ms']/c['wall_ms'],
            cuda_loss_trajectory_max_abs=max(abs(x-y) for x,y in zip(a['losses'],c['losses'])),
            torch_loss_trajectory_max_abs=max(abs(x-y) for x,y in zip(a['losses'],t['losses']))))
    saved_pairs={int(p['repeat']):p for p in csvrows('paired_repeat_comparisons.csv')}
    check('paired_results_recomputed',len(pairs)==len(saved_pairs)==3 and all(all(close(p[k],saved_pairs[p['repeat']][k]) for k in
           ('cuda_vs_native','cuda_vs_torch_matched_chunk','cuda_loss_trajectory_max_abs')) for p in pairs))
    cover=csvrows('coverage.csv')
    check('saved_coverage_matches',len(cover)==9 and {(r['method'],int(r['repeat'])) for r in cover}==expected and all(r['status']=='ok' for r in cover))
    write_csv(output/'matched_training.csv',summary)
    write_csv(output/'matched_repeats.csv',[{k:v for k,v in r.items() if k not in ('losses','selection','training_shape')} for r in rows])
    write_csv(output/'matched_comparisons.csv',pairs)
    write_csv(output/'matched_checks.csv',checks)
    by={s['method']:s for s in summary}; a,t,c=(by[m] for m in METHODS)
    metrics=dict(cuda_throughput_vs_native=c['tokens_per_s']/a['tokens_per_s'],cuda_throughput_vs_torch1024=c['tokens_per_s']/t['tokens_per_s'],
                 cuda_time_reduction_vs_native=1-c['wall_ms']/a['wall_ms'],cuda_memory_reduction_vs_native=1-c['peak_allocated_mib']/a['peak_allocated_mib'],
                 cuda_memory_reduction_vs_torch1024=1-c['peak_allocated_mib']/t['peak_allocated_mib'])
    evidence=dict(archive_sha256=hashlib.sha256(Path(archive).read_bytes()).hexdigest(),checks_total=len(checks),
        checks_passed=sum(x['passed'] for x in checks),all_checks_passed=all(x['passed'] for x in checks),
        source_sha256=env['source_sha256'],measured_trials=len(rows),measured_updates=sum(r['iterations'] for r in rows),
        actual_optimizer_updates=sum(r['optimizer_steps'] for r in rows),measured_tokens=sum(r['total_measured_tokens'] for r in rows),
        timed_seconds=sum(r['block_wall_s'] for r in rows),timed_block_range_s=[min(r['block_wall_s'] for r in rows),max(r['block_wall_s'] for r in rows)],
        max_preflight_gradient_error=max(v for r in nums for v in r['gradient_relative_errors']),
        max_loss_trajectory_difference=max(p['cuda_loss_trajectory_max_abs'] for p in pairs),metrics=metrics,
        scope='Offline integrity, arithmetic and recorded tolerance checks; not an independent GPU rerun. Initial and matched trials are not pooled.')
    (output/'matched_audit.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    if not evidence['all_checks_passed']:
        raise AssertionError('Failed follow-up checks: '+repr([x for x in checks if not x['passed']]))
    return evidence
