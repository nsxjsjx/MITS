import builtins,contextlib,io,json,time
from pathlib import Path
from revision.config import ROOT,dump,load_config,code_hashes
from revision.data import read_manifest

def local_config(path=None):
    c=load_config(path)
    # A copied project on Linux can use its own dataset/report without Windows paths.
    report_name=Path(str(c['report_root']).replace('\\','/')).name
    for key,suffix in [('data_root','dataset_expanded/'+c['data_version']),('report_root','pilot_reports/'+report_name),('source_encoder_local','external_datasets/models/graphcodebert-base')]:
        local=ROOT/suffix
        if not Path(c[key]).exists() and local.exists():c[key]=str(local)
    # Never silently send a new-version config back to the old dataset.
    if '\\' in c['control_root'] and not Path(c['control_root']).is_absolute():
        c['control_root']=str(ROOT/Path(c['control_root'].replace('\\','/')).name)
    return c

def session(mode,c):
    import uuid
    path=ROOT/'step13_results'/(time.strftime('%Y%m%d_%H%M%S')+'_'+mode+'_'+uuid.uuid4().hex[:6])
    path.mkdir(parents=True,exist_ok=False)
    dump(path/'base_config.json',c);dump(path/'code_hashes.json',code_hashes())
    return path

@contextlib.contextmanager
def forbid_test_reads(c):
    """Allow frozen manifest metadata, forbid test lists and all held-out feature/source files."""
    rows=read_manifest(c);test_ids={(r['task'],r['sample_id']) for r in rows if r['split']=='test'}
    root=Path(c['data_root']).resolve();real_builtin=builtins.open;real_io=io.open;attempts=[]
    def check(file):
        if not isinstance(file,(str,bytes,Path)):return
        try:p=Path(file).resolve();rel=p.relative_to(root)
        except (ValueError,TypeError):return
        parts=rel.parts
        denied='test' in parts or (parts and parts[-1].startswith('test_'))
        denied=denied or (len(parts)>2 and (parts[0],Path(parts[-1]).stem) in test_ids)
        if denied:attempts.append(str(p));raise AssertionError('Stage 13 attempted to open test data: '+str(p))
    def wrapped_builtin(file,*a,**kw):check(file);return real_builtin(file,*a,**kw)
    def wrapped_io(file,*a,**kw):check(file);return real_io(file,*a,**kw)
    builtins.open=wrapped_builtin;io.open=wrapped_io
    try:yield attempts
    finally:builtins.open=real_builtin;io.open=real_io

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
