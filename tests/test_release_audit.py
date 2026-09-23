"""CPU tests for the release audit and recorded results; fixtures are not new experiments."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile
import hashlib
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_initial import load_archive, close
from audit_followup import audit
from reproduce import main
from verify_release import verify

@pytest.fixture(scope='module')
def recomputed(tmp_path_factory):
    p=tmp_path_factory.mktemp('recompute')
    return main(p),p

def test_all_offline_checks(recomputed):
    r,_=recomputed
    assert r['all_checks_passed'] and r['checks_total']==64

def test_training_budget(recomputed):
    r,_=recomputed;m=r['matched_confirmation']
    assert m['measured_trials']==9 and m['measured_updates']==900
    assert m['actual_optimizer_updates']==1080 and m['measured_tokens']==1843200

def test_expected_scoped_effects(recomputed):
    r,_=recomputed;m=r['matched_confirmation']['metrics']
    assert close(m['cuda_throughput_vs_native'],1.14815444178033)
    assert close(m['cuda_throughput_vs_torch1024'],1.316186251120677)
    assert close(m['cuda_memory_reduction_vs_native'],.32269988212782774)

def test_datasets_are_not_pooled(recomputed):
    r,p=recomputed
    assert r['initial']['measured_training_updates']==180
    assert (p/'matched_training.csv').is_file() and (p/'initial_training.csv').is_file()

def test_bad_archive_path(tmp_path):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('run/config.json','{}');z.writestr('run/../escape','bad')
    with pytest.raises(ValueError,match='Unsafe'):load_archive(path)

def test_duplicated_archive_entry(tmp_path):
    path=tmp_path/'bad.zip'
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('run/config.json','{}')
        with pytest.warns(UserWarning):z.writestr('run/config.json','{}')
    with pytest.raises(ValueError,match='Duplicate'):load_archive(path)

def test_rehashed_wrong_measurement_rejected(tmp_path):
    files=load_archive(ROOT/'data/matched-confirmation.zip')
    name='raw/r0-native.json';r=json.loads(files[name]);r['tokens_per_s']*=2
    files[name]=json.dumps(r).encode()
    files['SHA256SUMS.json']=json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in files.items() if k!='SHA256SUMS.json'}).encode()
    zpath=tmp_path/'tampered.zip'
    with zipfile.ZipFile(zpath,'w') as z:
        for k,v in files.items():z.writestr('run/'+k,v)
    with pytest.raises(AssertionError,match='time_and_token_formulas'):
        audit(zpath,ROOT/'data/initial-study.zip',tmp_path/'out',ROOT)

def test_source_package_contains_no_latex():
    for zpath in (ROOT/'data').glob('*.zip'):
        with zipfile.ZipFile(zpath) as z:
            assert not any(Path(n).suffix in ('.tex','.bib','.aux','.log') for n in z.namelist())

def test_release_manifest_tamper(tmp_path):
    (tmp_path/'a.txt').write_text('a')
    (tmp_path/'SHA256SUMS.json').write_text(json.dumps({'a.txt':hashlib.sha256(b'a').hexdigest()}))
    assert verify(tmp_path)==1
    (tmp_path/'a.txt').write_text('b')
    with pytest.raises(ValueError,match='modified'):verify(tmp_path)

def test_notebook_schema_and_python():
    import nbformat
    nb=nbformat.read(ROOT/'notebooks/BudgetCE.ipynb',as_version=4)
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type=='code':
            assert not c.outputs and c.execution_count is None
            compile(c.source,'notebook-cell','exec')
