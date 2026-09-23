"""CPU-only summaries. Missing, failed and over-budget measurements remain explicit."""
import itertools
import math
import statistics
from pathlib import Path
import pandas as pd
from .common import read_json,write_json,plans,digest,MIB


def _rows(out,phase):return [read_json(p) for p in sorted((Path(out)/'raw'/phase).glob('*.json'))]

def _cv(values):
    return statistics.stdev(values)/statistics.mean(values) if len(values)>1 else None


def _summary(rows,expected):
    good=[r for r in rows if r['status']=='ok']
    complete=len(rows)==expected and len({r['repeat'] for r in rows})==expected and len(good)==expected
    d=dict(repeats_recorded=len(rows),repeats_expected=expected,valid_repeats=len(good),eligible=complete,
           statuses=';'.join(sorted({r['status'] for r in rows})) or 'missing')
    if good:
        # Block mean intervals; aggregate uses total block time / total repetitions.
        mean=sum(r['block_wall_s'] for r in good)*1000/sum(r['iterations'] for r in good)
        values=[r['wall_ms'] for r in good]
        d.update(wall_ms=mean,min_trial_ms=min(values),max_trial_ms=max(values),trial_cv=_cv(values),
                 peak_allocated_mib=max(r['peak_allocated_bytes'] for r in good)/MIB,
                 extra_peak_allocated_mib=max(r['extra_peak_allocated_bytes'] for r in good)/MIB,
                 peak_reserved_mib=max(r['peak_reserved_bytes'] for r in good)/MIB)
    return d


def analyze(out):
    out=Path(out);cfg=read_json(out/'config.json');dest=out/'analysis';dest.mkdir(exist_ok=True)
    eval_rows=_rows(out,'evaluation');train_rows=_rows(out,'training');cal_rows=_rows(out,'calibration')
    coverage=[];op=[];train=[]
    for shape,plan in itertools.product(cfg['evaluation_shapes'],plans(cfg)):
        rs=[r for r in eval_rows if r['shape']==shape and r['plan']==plan]
        d=dict(**shape,plan=plan,**_summary(rs,cfg['repeats']))
        if 'wall_ms' in d:d['tokens_per_s']=shape['n']*1000/d['wall_ms']
        op.append(d)
        for repeat in range(cfg['repeats']):
            found=[r for r in rs if r['repeat']==repeat]
            coverage.append(dict(stage='operator',condition=f'{shape}:{plan}',repeat=repeat,status=found[0]['status'] if len(found)==1 else 'missing_or_duplicate'))
    for seq,method in itertools.product(cfg['training_sequences'],['native','torch_chunked','adaptive']):
        rs=[r for r in train_rows if r['sequence']==seq and r['method']==method]
        d=dict(sequence=seq,method=method,**_summary(rs,cfg['training_repeats']))
        if rs:d['plan']=';'.join(sorted({r.get('plan','') for r in rs}))
        if 'wall_ms' in d:d['tokens_per_s']=seq*cfg['training_batch']*cfg['training_accumulation']*1000/d['wall_ms']
        train.append(d)
        for repeat in range(cfg['training_repeats']):
            found=[r for r in rs if r['repeat']==repeat]
            coverage.append(dict(stage='training',condition=f'S{seq}:{method}',repeat=repeat,status=found[0]['status'] if len(found)==1 else 'missing_or_duplicate'))
    pd.json_normalize(cal_rows).to_csv(dest/'calibration.csv',index=False)
    pd.json_normalize(eval_rows).to_csv(dest/'operator_trials.csv',index=False)
    pd.json_normalize(train_rows).to_csv(dest/'training_trials.csv',index=False)
    pd.DataFrame(coverage).to_csv(dest/'coverage.csv',index=False)
    pd.DataFrame(op).to_csv(dest/'operator_summary.csv',index=False)
    pd.DataFrame(train).to_csv(dest/'training_summary.csv',index=False)
    policy=[]
    planner=read_json(out/'planner.json') if (out/'planner.json').exists() else {}
    decisions=read_json(out/'decisions_frozen_before_evaluation.json') if (out/'decisions_frozen_before_evaluation.json').exists() else []
    for dec in decisions:
        shape=dec['shape'];b=dec['budget_mib'];sel=dec['selection']
        row=dict(**shape,budget_mib=b,selected=sel['plan'] if sel else None,status='no_plan_predicted')
        comparable=[d for d in op if all(d[k]==shape[k] for k in shape) and d['plan'].startswith('cuda:')]
        fully_measured=len(comparable)==len(cfg['chunks']) and all(d['eligible'] for d in comparable)
        row['all_candidate_repeats_valid']=fully_measured
        feasible=[d for d in comparable if d['eligible'] and d.get('extra_peak_allocated_mib',math.inf)<=b]
        if feasible and fully_measured:
            oracle=min(feasible,key=lambda x:x['wall_ms']);row.update(best_observed=oracle['plan'],best_observed_ms=oracle['wall_ms'])
        else:oracle=None
        if sel:
            r=next((d for d in comparable if d['plan']==sel['plan']),None)
            row.update(predicted_ms=sel['predicted_ms'],predicted_extra_mib=sel['predicted_extra_bytes']/MIB)
            if r and r['eligible']:
                ok=r['extra_peak_allocated_mib']<=b
                row.update(actual_ms=r['wall_ms'],actual_extra_mib=r['extra_peak_allocated_mib'],
                           budget_met=ok,status='ok' if ok else 'budget_violation')
                if oracle and ok:row['regret_vs_best_observed']=r['wall_ms']/oracle['wall_ms']-1
                fixed=next((d for d in feasible if d['plan']==f"cuda:{cfg['torch_chunk']}"),None)
                if fixed and ok:
                    delta=fixed['wall_ms']-r['wall_ms'];row['speedup_vs_fixed_cuda']=fixed['wall_ms']/r['wall_ms']
                    row['calibration_break_even_uses']=math.ceil(planner['calibration_cost_ms']/delta) if delta>0 else None
                pred_feasible=[p for p in dec['predictions'] if p['predicted_extra_bytes']<=b*MIB]
                largest=max(pred_feasible,key=lambda p:int(p['plan'].split(':')[1])) if pred_feasible else None
                if largest:
                    ld=next((d for d in comparable if d['plan']==largest['plan'] and d['eligible']),None)
                    row['largest_predicted_feasible']=largest['plan']
                    if ld:
                        row['largest_plan_budget_met']=ld['extra_peak_allocated_mib']<=b
                        if ok and row['largest_plan_budget_met']:row['speedup_vs_largest']=ld['wall_ms']/r['wall_ms']
            else:row['status']='selected_plan_missing_or_failed'
        policy.append(row)
    pd.DataFrame(policy).to_csv(dest/'policy_evaluation.csv',index=False)
    valid_op=sum(d['eligible'] for d in op);valid_train=sum(d['eligible'] for d in train)
    notes=f'''# BudgetCE results status\n\nValid complete operator conditions: {valid_op}/{len(op)}.\nValid complete training conditions: {valid_train}/{len(train)}.\n\nThis is {'a screening run' if cfg['profile']=='screen' else 'a bounded confirmation run'}, not proof of performance or quality outside the recorded settings.\n\n- Operator timing includes projection, loss and backward; no optimizer.\n- Training timing includes the full decoder and actual AdamW updates.\n- CUDA-event intervals include stream idle gaps, not just kernel-active time.\n- Operator budget is EXTRA PyTorch allocated memory above resident inputs, including gradient outputs.\n- Whole-training budget uses peak allocated memory, not the operator budget.\n- Reserved and allocated memory differ; NVML before/after readings are not peaks.\n- Selector is fit only on calibration token counts; held-out N values are not used to fit.\n- Regret uses a hindsight best-observed candidate, only when all candidate repeats are valid.\n- Ranges/CV are descriptive; no significance tests or confidence intervals.\n- Loss objective is dense CE; numerical tolerances do not prove convergence/accuracy preservation.\n- No vocabulary filtering; FP16 rounding and FP32 chunk-gradient accumulation are explicit.\n- Higher-order derivatives, class weights, label smoothing and torch.compile integration are not supported.\n\nRead coverage.csv and policy_evaluation.csv before reporting improvements. A budget violation is a result, not something to remove.\n'''
    (dest/'STATUS.md').write_text(notes)
    return dict(operator=op,training=train,policy=policy,coverage=coverage)


def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);d=analyze(out);dest=out/'analysis/figures';dest.mkdir(exist_ok=True)
    for n,v,w in sorted({(r['n'],r['v'],r['d']) for r in d['operator']}):
        rs=[r for r in d['operator'] if (r['n'],r['v'],r['d'])==(n,v,w) and r['eligible']]
        if not rs:continue
        fig,ax=plt.subplots(figsize=(7,4.5))
        for r in rs:
            ax.scatter(r['extra_peak_allocated_mib'],r['tokens_per_s'],label=r['plan'])
            ax.annotate(r['plan'],(r['extra_peak_allocated_mib'],r['tokens_per_s']),xytext=(4,4),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Extra peak allocated memory (MiB)',ylabel='Forward + backward tokens/s',title=f'N={n}, V={v}, D={w}')
        ax.grid(alpha=.2);fig.tight_layout();fig.savefig(dest/f'operator_N{n}_V{v}.png',dpi=160);plt.close(fig)
    for seq in sorted({r['sequence'] for r in d['training']}):
        rs=[r for r in d['training'] if r['sequence']==seq and r['eligible']]
        if not rs:continue
        fig,ax=plt.subplots(figsize=(6,4));ax.bar([r['method'] for r in rs],[r['tokens_per_s'] for r in rs])
        ax.set(ylabel='Whole-training tokens/s',title=f'Sequence length {seq}');fig.tight_layout()
        fig.savefig(dest/f'training_S{seq}.png',dpi=160);plt.close(fig)
    return dest
