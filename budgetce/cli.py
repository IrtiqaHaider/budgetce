import argparse
import hashlib
import json
import shutil
import sys
import traceback
import zipfile
from pathlib import Path
from .common import ROOT,initialize,read_json,write_json,make_config,plan_counts


def export(out):
    out=Path(out).resolve();dest=out/'source'
    # No model weights or private environment variables are exported.
    for folder in ['budgetce','configs','tests','docs','report','scripts']:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc','.so','.o'):continue
            rel=p.relative_to(ROOT);q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
    for name in ['README.md','requirements-colab.txt','LICENSE','REPRODUCIBILITY.md']:
        p=ROOT/name
        if p.exists():shutil.copyfile(p,dest/name)
    hashes={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.json' and not p.name.endswith('.tmp')}
    write_json(out/'SHA256SUMS.json',hashes)
    archive=out.parent/(out.name+'_results.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and not p.name.endswith('.tmp'):z.write(p,str(p.relative_to(out.parent)))
    print('Exported:',archive,flush=True)
    return archive


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['init','validate','operators','train','analyze','profile','export','all'])
    p.add_argument('--out',required=True);p.add_argument('--config');p.add_argument('--profile',default='confirm',choices=['screen','confirm'])
    a=p.parse_args();out=Path(a.out)
    cfg=read_json(a.config) if a.config else (read_json(out/'config.json') if (out/'config.json').exists() else make_config(a.profile))
    try:
        if a.stage not in ['analyze','export']:
            initialize(out,cfg)
        if a.stage=='init':print(json.dumps(plan_counts(cfg),indent=2))
        if a.stage in ['validate','all']:
            from .validate import validate_gpu
            validate_gpu(out)
        if a.stage in ['operators','all']:
            from .experiment import run_operators
            run_operators(out,cfg)
        if a.stage in ['train','all']:
            from .experiment import run_training
            run_training(out,cfg)
        if a.stage in ['analyze','all']:
            from .analyze import plots
            print('Plots:',plots(out))
        if a.stage=='profile' or (a.stage=='all' and cfg['run_profiler']):
            from .profile import traces
            traces(out,cfg)
        if a.stage in ['export','all']:export(out)
    except Exception as exc:
        write_json(out/'errors'/f'{a.stage}.json',dict(error=repr(exc),traceback=traceback.format_exc(),stage=a.stage))
        print(f'ERROR in {a.stage}: {exc}\nSaved full error to {out}/errors. Export this run for diagnosis.',file=sys.stderr,flush=True)
        raise

if __name__=='__main__':main()
