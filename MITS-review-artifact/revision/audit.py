import argparse,collections,csv,json,re
from pathlib import Path
from revision.config import ROOT,load_config,dump,digest
from revision.data import read_manifest,coverage_ids

def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def audit(c,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=read_manifest(c)
    stats=collections.Counter((r['task'],r['split'],int(r['label'])) for r in rows)
    write_csv(out/'class_counts.csv',[dict(task=t,split=s,label=y,count=n) for (t,s,y),n in sorted(stats.items())])
    fields=['source','label_basis','compile_status','compiler_version','pairing_evidence','training_only']
    counts=[]
    for field in fields:
        for (task,value),n in collections.Counter((r['task'],r.get(field) or 'UNKNOWN') for r in rows).items():counts.append(dict(task=task,field=field,value=value,count=n))
    write_csv(out/'provenance_counts.csv',counts)
    evidence=[]
    for r in rows:evidence.append(dict(task=r['task'],sample_id=r['sample_id'],source_file_exists=Path(r['source_file']).is_file(),
        target_contract=r['target_contract'],pairing_evidence=r['pairing_evidence'],explorer_verified='UNKNOWN',
        label_basis=r['label_basis'],label_evidence=r.get('label_evidence',''),annotators='UNKNOWN',agreement='UNKNOWN',unreachable_policy='UNKNOWN'))
    write_csv(out/'evidence_levels.csv',evidence)
    rr=Path(c['report_root']);flow={}
    for name in ['old_registry.json','new_registry.json']:
        records=json.loads((rr/name).read_text(encoding='utf-8-sig'))
        flow[name]={field:dict(collections.Counter(str(r.get(field,'UNKNOWN')) for r in records)) for field in ['compile_status','feature_status','decision','exclusion_reason']}
    dump(out/'processing_flow.json',flow)
    for fraction in [.1,.25,.5,1.]:
        ids=coverage_ids(rows,fraction,c['coverage_seed'])
        selected=[r for r in rows if r['task']+'/'+r['sample_id'] in ids]
        write_csv(out/f'coverage_{int(fraction*100)}.csv',[{k:r[k] for k in ['task','sample_id','label','split_group']} for r in selected])
        dump(out/f'coverage_{int(fraction*100)}_counts.json',{t:dict(selected=sum(r['task']==t for r in selected),total=sum(r['task']==t and r['split']=='train' for r in rows)) for t in c['tasks']})
    dump(out/'manifest_identity.json',{'sha256':digest(rr/'final_manifest.csv'),'count':len(rows),'split_seed':c['split_seed'],
         'unknowns':['historical compiler settings where not recorded','original annotator agreement','explorer verification coverage','historical disassembly failure denominator']})

def lengths(c,out):
    from transformers import RobertaTokenizer
    from pyevmasm import disassemble_all
    from revision.features import tokenizer
    tok=RobertaTokenizer.from_pretrained(c.get('source_encoder_local',c['source_encoder']),local_files_only=True)
    rows=read_manifest(c);results=[]
    for r in rows:
        d=Path(c['data_root'])/r['task'];sid=r['sample_id'];item={'task':r['task'],'sample_id':sid,'split':r['split']}
        try:
            code=(d/'sourcecode'/(sid+'.sol')).read_text(encoding='utf-8-sig')
            n=len(tok.encode(code,truncation=False));item.update(source_tokens=n,source_truncated=n>512,source_group='long' if n>512 else 'short')
            bc=(d/'bytecode_pure'/(sid+'.sol')).read_text().strip();instructions=list(disassemble_all(bytes.fromhex(bc)))
            # Exactly mirror the current opcode tokenizer's operand expansion.
            count=2+sum(1+(1 if isinstance(i.operand,int) else len(i.operand) if isinstance(i.operand,bytes) else 0) for i in instructions)
            item.update(bytecode_tokens=count,bytecode_truncated=count>512)
            gv=d/'cfg_gv'/(sid+'.gv')
            if gv.exists():
                # Parse this project's generated block labels, not arbitrary DOT.
                labels=re.findall(r'^\s*block_[^\s]+\s+\[label="((?:\\.|[^"\\])*)"',gv.read_text(encoding='utf-8-sig'),re.MULTILINE);ns=[]
                import torch
                saved=torch.load(d/'cfg_pyg_data'/(sid+'.pt'),map_location='cpu')
                if len(labels)!=len(saved.x):raise ValueError('CFG text node count differs from saved graph')
                for label in labels:
                    text=label.replace('\\l',' ').replace('\\n',' ')
                    words=[w for w in text.split() if not re.fullmatch('[0-9a-fA-F]+:',w)]
                    ns.append(len(words)+2)
                item.update(node_count=len(ns),node_max_tokens=max(ns) if ns else None,node_truncated_count=sum(n>50 for n in ns),node_lengths_status='gv_exact_token_count')
            else:item['node_lengths_status']='UNKNOWN_missing_original_CFG_text'
            item['status']='SUCCESS'
        except Exception as e:item.update(status='FAILED',error=str(e))
        results.append(item)
    write_csv(Path(out)/'lengths.csv',results)
    summary={}
    for t in c['tasks']:
        rr=[r for r in results if r['task']==t];summary[t]={'samples':len(rr),'failed':sum(r['status']!='SUCCESS' for r in rr)}
        for field in ['source_truncated','bytecode_truncated']:
            known=[r for r in rr if field in r];n=sum(r[field] for r in known)
            summary[t][field]={'count':n,'known_denominator':len(known),'ratio':n/len(known) if known else None}
        summary[t]['nodes']={'known':sum(r.get('node_count',0) for r in rr),'truncated':sum(r.get('node_truncated_count',0) for r in rr)}
    dump(Path(out)/'length_summary.json',summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--output',default=str(ROOT/'revision_reports'));p.add_argument('--lengths',action='store_true');a=p.parse_args();c=load_config(a.config);audit(c,a.output)
    if a.lengths:lengths(c,a.output)
