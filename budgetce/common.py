import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
MIB=1024**2

def digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,default=str).encode()).hexdigest()

def write_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False,default=str)+'\n')
    tmp.replace(path)

def read_json(path):return json.loads(Path(path).read_text())

def source_hash():
    paths=sorted([* (ROOT/'budgetce').rglob('*.py'),*(ROOT/'budgetce').rglob('*.cu')])
    return digest({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})

def configure_torch():
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.backends.cudnn.benchmark=False

def environment():
    if not torch.cuda.is_available():raise RuntimeError('Select Runtime > Change runtime type > NVIDIA GPU.')
    configure_torch()
    p=torch.cuda.get_device_properties(0)
    from torch.utils.cpp_extension import CUDA_HOME
    nvcc=Path(CUDA_HOME or '/missing')/'bin/nvcc'
    if not nvcc.is_file():raise RuntimeError('nvcc not found. CUDA toolkit is required; torch alone is insufficient.')
    cmd=lambda a:subprocess.check_output(a,text=True,stderr=subprocess.STDOUT).strip()
    return {'torch':str(torch.__version__),'torch_cuda':torch.version.cuda,
            'python':sys.version,'platform':platform.platform(),'gpu':p.name,
            'gpu_total_bytes':p.total_memory,'compute_capability':list(torch.cuda.get_device_capability(0)),
            'nvidia_smi':cmd(['nvidia-smi','--query-gpu=uuid,driver_version,name','--format=csv,noheader']),
            'nvcc':cmd([str(nvcc),'--version']),'cuda_home':str(CUDA_HOME),
            'source_sha256':source_hash(),'tf32':False,'fp16_reduced_precision_reduction':False,
            'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','matplotlib','ninja','pytest','nvidia-ml-py']},
            'cpu_threads':2,'attention_backend':'MATH; deliberately fixed, not maximum-performance attention'}


def initialize(out,cfg):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    env=environment()
    for name,obj in [('config.json',cfg),('environment.json',env)]:
        p=out/name
        if p.exists() and read_json(p)!=obj:
            raise RuntimeError(f'{name} changed. Use a new run name; do not mix hardware/software/configurations.')
        write_json(p,obj)
    return env


def require_gate(out):
    p=Path(out)/'validation/gpu.json'
    if not p.exists() or read_json(p).get('status')!='passed' or read_json(p).get('source_sha256')!=source_hash():
        raise RuntimeError('Run the GPU correctness gate first. No benchmark is allowed without it.')


def make_config(profile='confirm'):
    if profile not in ('screen','confirm'):raise ValueError('profile must be screen or confirm')
    cfg={
        'profile':profile,'seed':20260923,'compute_dtype':'float16','loss_scale':128.,
        'chunks':[64,256,1024],'torch_chunk':256,'warmup':1,'iterations':3,
        'repeats':3,'budgets_mib':[384,768],'planner_memory_margin':1.15,
        'calibration_shapes':[{'n':n,'v':v,'d':512} for n in [256,1024] for v in [32768,65536]],
        'evaluation_shapes':[{'n':n,'v':v,'d':512} for n in [512,2048,4096] for v in [32768,65536]],
        'training_sequences':[512,1024],'training_batch':1,'training_accumulation':2,
        'training_d':256,'training_layers':4,'training_heads':4,'training_vocab':32768,
        'training_warmup':2,'training_updates':10,'training_repeats':3,
        'training_budget_fraction':.85,'operator_selection_budget_mib':384,
        'optimizer':'AdamW','learning_rate':.001,'run_profiler':False,
    }
    if profile=='screen':
        cfg.update(chunks=[64,256],repeats=1,iterations=2,
                   calibration_shapes=[{'n':n,'v':32768,'d':512} for n in [256,1024]],
                   evaluation_shapes=[{'n':n,'v':32768,'d':512} for n in [512,2048]],
                   training_sequences=[512],training_updates=4,training_repeats=1)
    return cfg


def plans(cfg):return ['native',f"torch:{cfg['torch_chunk']}"]+[f'cuda:{c}' for c in cfg['chunks']]

def plan_counts(cfg):
    return {'calibration_blocks':len(cfg['calibration_shapes'])*len(plans(cfg)),
            'evaluation_blocks':len(cfg['evaluation_shapes'])*len(plans(cfg))*cfg['repeats'],
            'training_trials':len(cfg['training_sequences'])*3*cfg['training_repeats'],
            'training_measured_updates':len(cfg['training_sequences'])*3*cfg['training_repeats']*cfg['training_updates']}
