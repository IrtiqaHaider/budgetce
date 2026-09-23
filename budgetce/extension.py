"""JIT loader: keep Colab's PyTorch; build only this small CUDA extension."""
import hashlib
import os
from pathlib import Path
import torch

_LOADED = None

def load_extension(verbose: bool = False):
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the custom kernel; no CPU fallback is used.')
    from torch.utils.cpp_extension import load, CUDA_HOME
    if CUDA_HOME is None or not (Path(CUDA_HOME)/'bin/nvcc').is_file():
        raise RuntimeError('CUDA nvcc toolkit not found. Use a Colab NVIDIA GPU runtime, not CPU/TPU.')
    cap=torch.cuda.get_device_capability()
    if cap < (7,5):
        raise RuntimeError('This study targets compute capability >=7.5; older hardware is outside scope.')
    os.environ.setdefault('MAX_JOBS','1')
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{cap[0]}.{cap[1]}'
    src=Path(__file__).parent/'csrc/row_ce.cu'
    key=hashlib.sha256((src.read_text()+str(torch.__version__)+str(torch.version.cuda)+str(cap)).encode()).hexdigest()[:12]
    _LOADED=load(name='budgetce_'+key, sources=[str(src)],
                 extra_cuda_cflags=['-O2','-lineinfo'], extra_cflags=['-O2'],
                 verbose=verbose, with_cuda=True)
    return _LOADED
