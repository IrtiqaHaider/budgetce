"""Verify a result ZIP without executing bundled source or extracting archive paths."""
import argparse
import hashlib
import json
import math
import zipfile
from pathlib import PurePosixPath


def verify(path):
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(set(names))!=len(names):raise ValueError('Duplicate ZIP entries.')
        for name in names:
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts:raise ValueError('Unsafe archive path.')
        manifests=[n for n in names if n.endswith('/SHA256SUMS.json') and n.count('/')==1]
        if len(manifests)!=1:raise ValueError('Expected one top-level result hash manifest.')
        manifest=manifests[0];prefix=manifest.rsplit('/',1)[0]+'/'
        hashes=json.loads(z.read(manifest))
        covered={name[len(prefix):] for name in names if name.startswith(prefix) and not name.endswith('/') and name!=manifest}
        if covered!=set(hashes):raise ValueError('Uncovered or missing archive files.')
        for name,expected in hashes.items():
            if hashlib.sha256(z.read(prefix+name)).hexdigest()!=expected:raise ValueError('Hash mismatch: '+name)
        records=[]
        for name in names:
            if name.startswith(prefix+'raw/') and name.endswith('.json'):
                r=json.loads(z.read(name));records.append(r)
                if r['status']=='ok':
                    assert r['iterations']>0 and r['block_wall_s']>0
                    assert math.isclose(r['wall_ms'],1000*r['block_wall_s']/r['iterations'],rel_tol=1e-9)
                    assert r['peak_allocated_bytes']>=r['baseline_allocated_bytes']
                    assert r['extra_peak_allocated_bytes']==r['peak_allocated_bytes']-r['baseline_allocated_bytes']
                    if r['phase']=='training':
                        assert r['optimizer_steps']==r['expected_optimizer_steps']
                        expected_tps=r['total_measured_tokens']/r['block_wall_s']
                    else:expected_tps=r['shape']['n']*1000/r['wall_ms']
                    assert math.isclose(expected_tps,r['tokens_per_s'],rel_tol=1e-9)
        result={'hashed_files':len(hashes),'records':len(records),
                'statuses':{s:sum(r['status']==s for r in records) for s in sorted({r['status'] for r in records})},
                'scope':'file integrity and saved arithmetic only; not independent GPU reproduction'}
        print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('zip_path');verify(p.parse_args().zip_path)
