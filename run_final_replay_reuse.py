#!/usr/bin/env python3
"""Final 840-row replay with strict reuse of matching pinned diagnostic queries.

Run from the final-replay branch. Uses the exact diagnostic functions at their
original commits, not a separately reimplemented parameter formula.
"""
import argparse
import csv
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.environ['MPLBACKEND']='Agg'
ROOT=Path(__file__).resolve().parent
BASE='a1fd62e6e024ea5fff9a9897c4672823d467c96b'
VERSION='iotj-final-replay-reuse-v1'
FILES=[f'{e}_alpha{a}_runs.csv' for e in ('impact_epsilon_and_T','impact_topology') for a in (0,1)]
METRICS=('avg_regret_per_round','avg_loss_per_round','mse')
SPECS=[
    dict(branch='strong-pbdoo-radius-diagnostics',pin='4fbcf270f49a8cc97db944833fdc19198b02ea2b',
         module='diagnose_strong_pbdoo_radius',prefix='replay_strong_pbdoo_radius',
         alpha=1,algorithm='PBDOO-strong',candidate=0.08,field='multiplier',options='multipliers'),
    dict(branch='convex-vanilla-block-diagnostics',pin='ec4fd3560c6bcc28e919f862547575aaf48a6c5b',
         module='diagnose_convex_vanilla_block',prefix='replay_convex_vanilla_block',
         alpha=0,algorithm='PTF-PBDOO-fixed-block-optimized-eta-cor1',candidate='C',field='candidate',options='candidates'),
]

def require(ok,msg):
    if not ok:
        raise RuntimeError(msg)

def git(*args,cwd=ROOT):
    return subprocess.check_output(['git',*args],cwd=cwd).decode().strip()

def git_bytes(*args,cwd=ROOT):
    return subprocess.check_output(['git',*args],cwd=cwd,stderr=subprocess.DEVNULL)

def sha(b):
    return hashlib.sha256(b).hexdigest()

def variants(b):
    n=b.replace(b'\r\n',b'\n')
    return {sha(b),sha(n),sha(n.replace(b'\n',b'\r\n'))}

def read_csv(b):
    return list(csv.DictReader(io.StringIO(b.decode('utf-8-sig'),newline='')))

def write_csv(path,rows):
    require(bool(rows),f'Empty CSV: {path}')
    tmp=path.with_suffix('.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    tmp.replace(path)

def atomic(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10)

def identity(r):
    return '|'.join(map(str,(r['experiment'],r['algorithm'],int(float(r['alpha_reg'])),
                    int(r['T']),int(r['num_nodes']),r['topology'],float(r['epsilon']),
                    int(r['run']),int(r['seed']))))

def config(r):
    return int(r['T']),int(r['num_nodes']),r['topology'],float(r['epsilon'])

def diagnostic_key(r,c):
    T,n,top,eps=config(r)
    return f'{T}|{n}|{top}|{eps}|{c}|{r["run"]}|{r["seed"]}'

def checked_metrics(v,n):
    z={m:float(v[m]) for m in METRICS}
    require(all(math.isfinite(x) for x in z.values()),'Nonfinite metric')
    if 'normalized_regret' in v:
        require(close(v['normalized_regret'],z['avg_regret_per_round']/n),'Wrong normalized regret')
    return z

def load_base():
    src=ROOT/'replay_convex_pbdoo_c'/'complete'
    rows={}
    for f,count in zip(FILES,(700,700,840,840)):
        b=(src/f).read_bytes()
        original=git_bytes('show',f'{BASE}:replay_convex_pbdoo_c/complete/{f}')
        require(read_csv(b)==read_csv(original),f'Base archive changed: {f}')
        rows[f]=read_csv(b)
        require(len(rows[f])==count,f'Wrong row count: {f}')
        # BDOO epsilon is blank in one experiment; use full identity without float conversion.
        ids=[tuple(r[k] for k in ('experiment','algorithm','alpha_reg','T','num_nodes','topology','epsilon','run','seed')) for r in rows[f]]
        require(len(set(ids))==count,f'Duplicate rows: {f}')
        groups={}
        for r in rows[f]:
            k=tuple(r[x] for x in ('experiment','algorithm','alpha_reg','T','num_nodes','topology','epsilon'))
            groups.setdefault(k,[]).append(int(r['run']))
        require(all(sorted(v)==list(range(10)) for v in groups.values()),f'Wrong seed-index coverage: {f}')
    require(sum(map(len,rows.values()))==3080,'Wrong total')
    require(sum(r['algorithm']=='PBDOO-cor1' and r['parameter_version']=='convex-pbdoo-c-v1' for rr in rows.values() for r in rr)==420,'Convex C archive missing')
    return rows

def target_rows(rows,spec):
    target={}
    for f,rr in rows.items():
        for r in rr:
            if r['algorithm']==spec['algorithm']:
                require(int(float(r['alpha_reg']))==spec['alpha'] and r['parameter_version']=='final-tuned-v1','Wrong target archive')
                k=identity(r);require(k not in target,'Duplicate target');target[k]=r
    require(len(target)==420,'Wrong target count')
    return target

def setup(workspace,spec):
    branch,pin=spec['branch'],spec['pin']
    git('fetch','origin',f'refs/heads/{branch}')
    head=git('rev-parse','FETCH_HEAD')
    require(subprocess.run(['git','merge-base','--is-ancestor',pin,head],cwd=ROOT).returncode==0,'Diagnostic history diverged')
    changes=git('diff','--name-only',pin,head).splitlines()
    require(all(x.startswith(spec['prefix']+'_topology/') for x in changes),
            f'{branch}: post-pin changes are not limited to topology outputs; inspect before reuse')
    work=workspace/branch
    if not work.exists():
        git('worktree','add','--detach',str(work),pin)
    local_head=git('rev-parse','HEAD',cwd=work)
    require(subprocess.run(['git','merge-base','--is-ancestor',pin,local_head],cwd=work).returncode==0,'Local worktree diverged')
    local_changes=git('diff','--name-only',pin,local_head,cwd=work).splitlines()
    require(all(x.startswith(spec['prefix']+'_topology/') for x in local_changes) and not git('status','--porcelain','--untracked-files=no',cwd=work),
            f'{work}: tracked code must match the pinned diagnosis')
    return work,head

def materialize_sources(work,head,spec,stage):
    # Local topology results may be unpushed; prefer complete committed output
    # if available. Never read an actively running checkpoint.
    provenance={}
    for scope,suffix in [('epsilon','eps'),('topology','topology')]:
        dirname=spec['prefix']+'_'+suffix
        dest=stage/scope
        dest.mkdir(parents=True,exist_ok=True)
        names=('manifest.json','checkpoint.json','runs.csv','parameter_validation.csv','archive_reproduction.csv')
        remote={}
        for n in names:
            try:
                remote[n]=git_bytes('show',f'{head}:{dirname}/{n}')
            except subprocess.CalledProcessError:
                pass
        local={n:(work/dirname/n).read_bytes() for n in names if (work/dirname/n).is_file()}
        chosen=remote if 'runs.csv' in remote else local if 'manifest.json' in local else remote
        if scope=='epsilon':
            require(set(names)<=set(chosen),'Missing completed epsilon diagnosis')
        if not chosen:
            continue
        provenance[scope]={'kind':'remote' if chosen is remote else 'local','commit':head if chosen is remote else git('rev-parse','HEAD',cwd=work),'directory':dirname}
        require({'manifest.json','checkpoint.json','parameter_validation.csv','archive_reproduction.csv'}<=set(chosen),
                f'Incomplete provenance for {dirname}; finish or stop diagnostic cleanly before replay')
        for n,b in chosen.items():
            p=dest/n
            if p.exists():
                require(p.read_bytes()==b,f'Staged source changed: {p}; use fresh output or original source')
            else:
                p.write_bytes(b)
        # Avoid silently switching between staged and unstaged source state.
        require({p.name for p in dest.iterdir()}==set(chosen),f'Staged source file set changed: {dest}')
    provenance_path=stage/'provenance.json'
    if provenance_path.exists():
        require(json.loads(provenance_path.read_text())==provenance,'Source provenance changed; use fresh output')
    else:
        atomic(provenance_path,provenance)

def load_module(work,spec):
    sys.path.insert(0,str(work))
    return importlib.import_module(spec['module'])

def run_task(payload):
    work,spec,k,r,p=payload
    mod=load_module(Path(work),spec)
    returned,v=mod.run((k,r,p))
    require(returned==k,'Worker returned wrong key')
    return k,checked_metrics(v,int(r['num_nodes']))

def verified_import(work,spec,mod,target,params,stage):
    import numpy as np
    imported={}
    audit=[]
    for scope,suffix in [('epsilon','eps'),('topology','topology')]:
        src=stage/scope
        if not (src/'manifest.json').exists():
            continue
        m=json.loads((src/'manifest.json').read_text())
        exp='impact_epsilon_and_T' if scope=='epsilon' else 'impact_topology'
        f=f'{exp}_alpha{spec["alpha"]}_runs.csv'
        archive=work/'replay_convex_pbdoo_c'/'complete'/f
        require(m['scope']==scope and m['version']==mod.VERSION and spec['candidate'] in m[spec['options']],f'Wrong diagnosis manifest: {src}')
        require(m['python']==sys.version and m['numpy']==np.__version__,'Python/NumPy differs from diagnostic; use original environment')
        require(m['archive_sha256'] in variants(archive.read_bytes()),'Archive hash mismatch')
        require(m['dataset_sha256'] in variants(mod.DATA.read_bytes()),'Dataset hash mismatch')
        require(all(h in variants((work/p).read_bytes()) for p,h in m['implementation'].items()),'Diagnostic implementation hash mismatch')
        require(read_csv(archive.read_bytes())==read_csv(git_bytes('show',f'{BASE}:replay_convex_pbdoo_c/complete/{f}')),'Diagnostic and final archive differ')
        checks=read_csv((src/'archive_reproduction.csv').read_bytes())
        require(len(checks)==3 and all(r['passed']=='True' and math.isclose(float(r['archived_regret']),float(r['reproduced_regret']),rel_tol=1e-8,abs_tol=1e-8) for r in checks),'Original parameter reproduction failed')
        old=[r for r in target.values() if r['experiment']==exp and int(r['T']) in mod.HORIZONS and int(r['run']) in (0,1,2)]
        require(len(old)==27,'Unexpected diagnostic configuration count')
        allkeys={diagnostic_key(r,c) for r in old for c in m[spec['options']]}
        cp=json.loads((src/'checkpoint.json').read_text())
        require(len(allkeys)==m['count'] and set(cp)<=allkeys,'Unknown diagnostic checkpoint keys')
        tables=read_csv((src/'parameter_validation.csv').read_bytes())
        def same_candidate(x):
            return str(x)==str(spec['candidate']) if spec['alpha']==0 else close(x,spec['candidate'])
        ptable={config(r):r for r in tables if same_candidate(r[spec['field']])}
        require(len(ptable)==9,'Missing candidate parameter rows')
        for r in old:
            p=params[config(r)]
            q=ptable[config(r)]
            for name,value in p.items():
                require(name in q and (str(value)==q[name] if name=='cap_active' else close(value,q[name])),f'Parameter mismatch {name}: {config(r)}')
        final={}
        if (src/'runs.csv').exists():
            rr=read_csv((src/'runs.csv').read_bytes())
            require(len(rr)==m['count']==len(cp),'Completed diagnosis row count mismatch')
            for r in rr:
                c=float(r[spec['field']]) if spec['alpha'] else r[spec['field']]
                k=diagnostic_key(r,c)
                require(k in cp and k not in final,'Wrong/duplicate diagnostic result key')
                z=checked_metrics(r,int(r['num_nodes']))
                v=checked_metrics(cp[k],int(r['num_nodes']))
                require(all(close(z[x],v[x]) for x in METRICS),'CSV/checkpoint mismatch')
                final[k]=r
        elif scope=='epsilon':
            raise RuntimeError('Epsilon diagnosis must be complete')
        for r in old:
            dk=diagnostic_key(r,spec['candidate'])
            if dk not in cp:
                continue
            k=identity(r)
            z=checked_metrics(cp[dk],int(r['num_nodes']))
            if dk in final:
                fr=final[dk]
                require(identity(fr)==k and close(fr['archived_normalized_regret'],float(r['avg_regret_per_round'])/int(r['num_nodes'])),'Diagnostic source row mismatch')
            require(k not in imported,'Duplicate reused record')
            imported[k]=z
            audit.append(dict(identity=k,scope=scope,source=spec['prefix']+'_'+suffix,code_commit=spec['pin'],source_commit=json.loads((stage/'provenance.json').read_text())[scope]['commit'],
                              source_manifest_sha256=sha((src/'manifest.json').read_bytes()),
                              source_checkpoint_sha256=sha((src/'checkpoint.json').read_bytes()),
                              **z))
    require(sum(r['scope']=='epsilon' for r in audit)==27,'Expected 27 reusable epsilon candidate rows')
    return imported,audit

def worker(args):
    work=Path(args.worktree).resolve()
    spec=SPECS[args.internal_worker]
    mod=load_module(work,spec)
    import numpy as np
    out=Path(args.output).resolve()
    rows=load_base()
    target=target_rows(rows,spec)
    X,y=mod._load_bodyfat_dataset(mod.DATA)
    params={c:mod.params(X,y,*c,spec['candidate']) for c in sorted({config(r) for r in target.values()})}
    require(len(params)==42,'Wrong formal configuration count')
    write_csv(out/'parameter_validation.csv',[dict(T=c[0],num_nodes=c[1],topology=c[2],epsilon=c[3],**p) for c,p in params.items()])
    imported,audit=verified_import(work,spec,mod,target,params,out/'sources')
    write_csv(out/'reuse_audit.csv',audit)
    # Bind the entire tracked code snapshot, archive, environment and imported records.
    manifest=dict(version=VERSION,base=BASE,pin=spec['pin'],algorithm=spec['algorithm'],candidate=spec['candidate'],
                  python=sys.version,numpy=np.__version__,runner_sha256=sha(Path(__file__).read_bytes()),
                  sources={str(p.relative_to(out)):sha(p.read_bytes()) for p in sorted((out/'sources').rglob('*')) if p.is_file()},
                  target_sha256=sha(json.dumps(target,sort_keys=True).encode()),reuse_count=len(imported))
    mp=out/'manifest.json'; cp=out/'checkpoint.json'
    if mp.exists():
        require(json.loads(mp.read_text())==manifest,'Formal inputs changed; do not overwrite checkpoint')
    else:
        require(not cp.exists(),'Checkpoint without manifest');atomic(mp,manifest)
    done=json.loads(cp.read_text()) if cp.exists() else {}
    require(set(done)<=set(target),'Unknown formal checkpoint key')
    for k,v in done.items():
        checked_metrics(v,int(target[k]['num_nodes']))
    for k,v in imported.items():
        require(k not in done or all(close(done[k][m],v[m]) for m in METRICS),'Formal checkpoint differs from imported data')
        done[k]=v
    atomic(cp,done)
    print(f'{spec["algorithm"]}: total=420 reused={len(imported)} completed={len(done)} remaining={420-len(done)}',flush=True)
    if args.validate_only:
        return
    pending=[(str(work),spec,k,r,params[config(r)]) for k,r in target.items() if k not in done]
    pending.sort(key=lambda t:int(t[3]['T'])*int(t[3]['num_nodes']),reverse=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_task,t):t[2] for t in pending}
        for f in as_completed(futures):
            k,v=f.result()
            require(k==futures[f] and k not in done,'Unexpected worker key')
            done[k]=v
            atomic(cp,done)
            print(f'{spec["algorithm"]} {len(done)}/420 {k}',flush=True)
    require(len(done)==420,'Incomplete formal replay')
    require(not git('status','--porcelain','--untracked-files=no',cwd=work),'Tracked diagnostic implementation changed')
    atomic(out/'results.json',done)

def merge(rows,out):
    all_results={}
    for s in SPECS:
        d=out/s['prefix']
        result=json.loads((d/'results.json').read_text())
        require(set(result)==set(target_rows(rows,s)),'Wrong result coverage')
        require(not (set(result)&set(all_results)),'Duplicate result identity')
        all_results.update(result)
    require(len(all_results)==840,'Wrong replacement count')
    dest=out/'complete';dest.mkdir(exist_ok=True)
    counts=Counter()
    for f,rr in rows.items():
        merged=[]
        for r in rr:
            new=dict(r)
            if r['algorithm'] in {s['algorithm'] for s in SPECS}:
                k=identity(r)
                new.update(checked_metrics(all_results[k],int(r['num_nodes'])))
                new['implementation']=new['parameter_version']=VERSION
                counts['replaced']+=1
            else:
                counts['unchanged']+=1
            merged.append(new)
        write_csv(dest/f,merged)
        actual=read_csv((dest/f).read_bytes())
        for old,new in zip(rr,actual):
            if old['algorithm'] in {s['algorithm'] for s in SPECS}:
                require(all(old[k]==new[k] for k in old if k not in (*METRICS,'implementation','parameter_version')),'Changed original identity')
            else:
                require(old==new,'Changed frozen row')
    require(counts=={'replaced':840,'unchanged':2240},'Wrong merge count')
    import adv_setting.adv_setting_common as C
    from final_tuned_support import compare_results
    for f in FILES:
        errors=C.validate_run_rows(read_csv((dest/f).read_bytes()),expected_runs=10,base_seed=20261109)
        require(not errors,f'{f}: {errors[:5]}')
        C.recompute_summary_only(dest/f,dest/f.replace('_runs.csv','_summary.csv'))
    compare_results(ROOT/'replay_convex_pbdoo_c'/'complete',dest,out/'comparison',FILES)
    subprocess.run([sys.executable,str(ROOT/'redraw_figures.py'),str(dest),str(out/'figures')],cwd=ROOT,check=True)
    reused=sum(json.loads((out/s['prefix']/'manifest.json').read_text())['reuse_count'] for s in SPECS)
    atomic(out/'final_audit.json',dict(total=3080,unchanged=2240,replaced=840,reused=reused,newly_computed=840-reused))
    print(f'PASS: 3080 rows; 2240 unchanged; 840 target rows, {reused} reused, {840-reused} newly computed.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=ROOT.parent/(ROOT.name+'_topology_validation'))
    p.add_argument('--output',type=Path,default=ROOT/'replay_iotj_final_reuse')
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--internal-worker',type=int,choices=(0,1),help=argparse.SUPPRESS)
    p.add_argument('--worktree',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.internal_worker is not None:
        worker(args);return
    out=args.output.resolve();workspace=args.workspace.resolve()
    require(out!=ROOT and out!=workspace and workspace not in out.parents and out not in workspace.parents,'Separate output and worktree directories required')
    require(out!=ROOT/'replay_convex_pbdoo_c' and ROOT/'replay_convex_pbdoo_c' not in out.parents,'Never overwrite base archive')
    rows=load_base()
    out.mkdir(parents=True,exist_ok=True);workspace.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for i,s in enumerate(SPECS):
        work,head=setup(workspace,s)
        d=out/s['prefix'];d.mkdir(exist_ok=True)
        materialize_sources(work,head,s,d/'sources')
        jobs.append([sys.executable,'-u',str(Path(__file__).resolve()),'--internal-worker',str(i),'--worktree',str(work),'--output',str(d),'--workers',str(args.workers)])
    for cmd in jobs:
        subprocess.run(cmd+['--validate-only'],cwd=ROOT,check=True)
    if args.validate_only:
        print('Preflight complete; no new simulations executed. Review reuse counts before running.')
        return
    for cmd in jobs:
        subprocess.run(cmd,cwd=ROOT,check=True)
    merge(rows,out)

if __name__=='__main__':
    main()
