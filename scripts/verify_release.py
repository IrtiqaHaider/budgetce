#!/usr/bin/env python3
"""Verify every immutable release file listed in SHA256SUMS.json (CPU only)."""
import hashlib
import json
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[1]

def verify(root=ROOT):
    root=Path(root);entries=json.loads((root/'SHA256SUMS.json').read_text())
    for name,sha in entries.items():
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Invalid manifest path')
        f=root/name
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=sha:
            raise ValueError('Missing or modified release file: '+name)
    print(f'Verified {len(entries)} release files. Extra local files are not covered.')
    return len(entries)
if __name__=='__main__':verify()
