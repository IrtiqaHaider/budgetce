import numpy as np
import pytest
from budgetce.planner import Planner,workspace_estimate,nonnegative_fit
from budgetce.common import make_config,plan_counts


def training_rows():
    return [dict(status='ok',plan=f'cuda:{c}',shape=dict(n=n,v=32768,d=512),wall_ms=(n/1000+c/10000),
                 extra_peak_allocated_bytes=workspace_estimate(dict(n=n,v=32768,d=512),f'cuda:{c}')*.8,
                 calibration_total_ms=20) for c in [64,256] for n in [256,1024]]

def test_fit_nonnegative():
    x=np.array([[1,0,2],[1,2,1],[1,3,4],[1,4,9]])
    b=np.array([2,3,4]);out=nonnegative_fit(x,x@b)
    np.testing.assert_allclose(out,b,rtol=1e-8)

def test_budget_rejection():
    p=Planner.fit(training_rows())
    assert p.select(dict(n=2048,v=32768,d=512),1) is None

def test_roundtrip_and_positive_predictions():
    p=Planner.fit(training_rows());q=Planner.from_dict(p.as_dict())
    shape=dict(n=2048,v=32768,d=512)
    assert p.select(shape,2**30)==q.select(shape,2**30)
    assert all(x['predicted_ms']>0 for x in p.predictions(shape))

def test_memory_increases_with_chunk():
    shape=dict(n=4096,v=65536,d=512)
    assert workspace_estimate(shape,'cuda:64')<workspace_estimate(shape,'cuda:256')

def test_heldout_token_counts_disjoint():
    for profile in ['screen','confirm']:
        c=make_config(profile)
        assert not {s['n'] for s in c['calibration_shapes']} & {s['n'] for s in c['evaluation_shapes']}

def test_budget_counts():
    assert plan_counts(make_config())==dict(calibration_blocks=20,evaluation_blocks=90,training_trials=18,training_measured_updates=180)
    assert plan_counts(make_config('screen'))==dict(calibration_blocks=8,evaluation_blocks=8,training_trials=3,training_measured_updates=12)

def test_no_cuda_calibration_is_error():
    with pytest.raises(RuntimeError):Planner.fit([])
