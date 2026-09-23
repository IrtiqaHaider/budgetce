"""Small nonnegative cost model. Fit only calibration shapes; no online test-shape tuning."""
import itertools
import math
import numpy as np


def chunk_size(plan,n):
    return n if plan=='native' else min(n,int(plan.split(':')[1]))


def features(shape,plan):
    n,v,d=(shape[k] for k in ('n','v','d'))
    return np.array([1.,n*v*d/1e9,math.ceil(n/chunk_size(plan,n))],dtype=float)


def workspace_estimate(shape,plan):
    """Heuristic of extra allocator bytes, NOT a proven bound or total training memory.

    Includes casts, gradient outputs/accumulators and a GEMM dW temporary. Different
    CE paths have different chunk-tensor terms; allocator/library effects are
    calibrated and budget violations remain visible on held-out measurements.
    """
    n,v,d=(shape[k] for k in ('n','v','d'));m=chunk_size(plan,n)
    ce_bytes=2 if plan.startswith('cuda:') else 12
    return int(8*v*d+8*n*d+ce_bytes*m*v+32*m+4096)


def nonnegative_fit(x,y):
    """Exact active-set enumeration for three nonnegative coefficients."""
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    best=np.zeros(x.shape[1]);err=float(y@y)
    for k in range(1,x.shape[1]+1):
        for cols in itertools.combinations(range(x.shape[1]),k):
            b=np.linalg.lstsq(x[:,cols],y,rcond=None)[0]
            if (b < -1e-10).any():continue
            candidate=np.zeros(x.shape[1]);candidate[list(cols)]=np.maximum(b,0)
            score=float(np.sum((x@candidate-y)**2))
            if score<err:best,err=candidate,score
    return best


class Planner:
    def __init__(self,models,calibration_cost_ms=0.,safety=1.15):
        self.models=models;self.calibration_cost_ms=float(calibration_cost_ms);self.safety=safety

    @classmethod
    def fit(cls,rows,safety=1.15):
        usable=[r for r in rows if r['status']=='ok' and r['plan'].startswith('cuda:')]
        models={}
        for plan in sorted({r['plan'] for r in usable}):
            rs=[r for r in usable if r['plan']==plan]
            x=np.stack([features(r['shape'],plan) for r in rs]);y=np.array([r['wall_ms'] for r in rs])
            models[plan]={
                'coefficients':nonnegative_fit(x,y).tolist(),
                'memory_multiplier':float(max(r['extra_peak_allocated_bytes']/workspace_estimate(r['shape'],plan) for r in rs)*safety),
                'calibration_shapes':[r['shape'] for r in rs],
                'fit_rmse_ms':float(np.sqrt(np.mean((x@nonnegative_fit(x,y)-y)**2))),
            }
        if not models:raise RuntimeError('No successful CUDA calibration rows; cannot build a planner.')
        return cls(models,sum(float(r.get('calibration_total_ms',0)) for r in rows),safety)

    def predictions(self,shape):
        out=[]
        for plan,model in self.models.items():
            out.append({'plan':plan,'predicted_ms':max(1e-9,float(features(shape,plan)@np.array(model['coefficients']))),
                        'predicted_extra_bytes':int(math.ceil(workspace_estimate(shape,plan)*model['memory_multiplier']))})
        return out

    def select(self,shape,budget_bytes):
        if budget_bytes<=0:raise ValueError('A positive EXTRA allocated-memory budget is required.')
        viable=[p for p in self.predictions(shape) if p['predicted_extra_bytes']<=budget_bytes]
        if not viable:return None
        return min(viable,key=lambda x:(x['predicted_ms'],x['plan']))

    def as_dict(self):
        return {'models':self.models,'calibration_cost_ms':self.calibration_cost_ms,'safety':self.safety,
                'scope':'CUDA chunk selection; calibration shapes only; predicted budget, not a hard allocator limit'}

    @classmethod
    def from_dict(cls,d):return cls(d['models'],d['calibration_cost_ms'],d['safety'])
