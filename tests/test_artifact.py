import json
from pathlib import Path
import pytest
import torch
from budgetce.common import make_config,write_json,read_json,require_gate,digest
from budgetce.analyze import analyze
from budgetce.experiment import checkpoint


def fake_trial(shape,plan,repeat,status='ok',peak=100):
    return dict(shape=shape,plan=plan,repeat=repeat,status=status,block_wall_s=.006,iterations=3,
                wall_ms=2.,peak_allocated_bytes=peak,extra_peak_allocated_bytes=peak,
                peak_reserved_bytes=peak,phase='evaluation')

def test_missing_not_success(tmp_path):
    write_json(tmp_path/'config.json',make_config('screen'))
    result=analyze(tmp_path)
    assert not any(x['eligible'] for x in result['operator'])
    assert all(x['status']=='missing_or_duplicate' for x in result['coverage'])

def test_failed_repeat_excluded(tmp_path):
    cfg=make_config('screen');cfg['repeats']=2
    write_json(tmp_path/'config.json',cfg);shape=cfg['evaluation_shapes'][0]
    for k,status in enumerate(['ok','oom']):
        write_json(tmp_path/f'raw/evaluation/{k}.json',fake_trial(shape,'native',k,status))
    s=analyze(tmp_path)['operator'][0]
    assert s['valid_repeats']==1 and not s['eligible']

def test_policy_budget_violation_retained(tmp_path):
    cfg=make_config('screen');write_json(tmp_path/'config.json',cfg);shape=cfg['evaluation_shapes'][0]
    for p in ['cuda:64','cuda:256']:
        r=fake_trial(shape,p,0,peak=600*1024**2 if p.endswith('256') else 200*1024**2)
        write_json(tmp_path/f"raw/evaluation/{p.replace(':','')}.json",r)
    dec=dict(shape=shape,budget_mib=384,selection=dict(plan='cuda:256',predicted_ms=2,predicted_extra_bytes=300*1024**2),predictions=[])
    write_json(tmp_path/'decisions_frozen_before_evaluation.json',[dec])
    result=analyze(tmp_path)['policy'][0]
    assert result['status']=='budget_violation' and not result['budget_met']
    assert 'regret_vs_best_observed' not in result

def test_checkpoints_not_retried(tmp_path):
    count=[0]
    def fn():count[0]+=1;return {'metric':5}
    meta={'x':1}
    checkpoint(tmp_path,'test',meta,fn);checkpoint(tmp_path,'test',meta,fn)
    assert count==[1]

def test_error_saved_then_blocks_retry(tmp_path):
    count=[0]
    def fn():count[0]+=1;raise ValueError('test failure')
    with pytest.raises(ValueError):checkpoint(tmp_path,'test',{'a':1},fn)
    with pytest.raises(RuntimeError,match='Prior failed'):checkpoint(tmp_path,'test',{'a':1},fn)
    assert count==[1]

def test_no_gpu_gate_no_benchmark(tmp_path):
    with pytest.raises(RuntimeError):require_gate(tmp_path)

def test_nonfinite_json_rejected(tmp_path):
    with pytest.raises(ValueError):write_json(tmp_path/'bad.json',{'x':float('nan')})

def test_json_roundtrip(tmp_path):
    value={'a':[1,2],'b':None};write_json(tmp_path/'a.json',value)
    assert read_json(tmp_path/'a.json')==value

def test_export_verification_and_tamper(tmp_path):
    import importlib.util,zipfile,hashlib
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('export_check',root/'scripts/verify_export.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    payload=b'{"test":"synthetic fixture, not benchmark evidence"}'
    path=tmp_path/'bundle.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('run/test.json',payload)
        z.writestr('run/SHA256SUMS.json',json.dumps({'test.json':hashlib.sha256(payload).hexdigest()}))
    result=module.verify(path)
    assert result['hashed_files']==1 and result['records']==0
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('run/test.json',payload+b' ')
        z.writestr('run/SHA256SUMS.json',json.dumps({'test.json':hashlib.sha256(payload).hexdigest()}))
    with pytest.raises(ValueError,match='Hash mismatch'):module.verify(path)
