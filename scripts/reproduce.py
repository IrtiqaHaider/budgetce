#!/usr/bin/env python3
"""Recompute all published tables from the packaged evidence; CPU only."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from audit_initial import audit as audit_initial
from audit_followup import audit as audit_followup

ROOT=Path(__file__).resolve().parents[1]

def main(output=None):
    out=Path(output or ROOT/'results');out.mkdir(parents=True,exist_ok=True)
    prov=json.loads((ROOT/'data/provenance.json').read_text())
    for e in prov['inputs']:
        if hashlib.sha256((ROOT/e['packaged_file']).read_bytes()).hexdigest()!=e['packaged_sha256']:
            raise ValueError('Evidence archive differs from release provenance: '+e['packaged_file'])
    with tempfile.TemporaryDirectory() as td:
        with contextlib.redirect_stdout(io.StringIO()):
            initial=audit_initial(ROOT/'data/initial-study.zip',Path(td))
        for old,new in {'operator_recomputed.csv':'operator_summary.csv','operator_comparisons.csv':'operator_comparisons.csv',
            'training_recomputed.csv':'initial_training.csv','training_comparisons.csv':'initial_training_comparisons.csv',
            'training_repeat_ratios.csv':'initial_repeat_comparisons.csv','policy_recomputed.csv':'policy_evaluation.csv',
            'audit_checks.csv':'initial_checks.csv','audit_evidence.json':'initial_audit.json'}.items():
            shutil.copyfile(Path(td)/old,out/new)
    matched=audit_followup(ROOT/'data/matched-confirmation.zip',ROOT/'data/initial-study.zip',out,ROOT)
    result={'initial':initial,'matched_confirmation':matched,'scope':'Offline audit; no GPU benchmark executed.',
        'all_checks_passed':initial['all_checks_passed'] and matched['all_checks_passed'],
        'checks_passed':initial['checks_passed']+matched['checks_passed'],
        'checks_total':initial['checks_total']+matched['checks_total']}
    (out/'validation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f"Verified {result['checks_passed']}/{result['checks_total']} offline checks.")
    print('Initial and confirmation measurements kept separate. Tables:',out)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path)
    main(p.parse_args().out)
