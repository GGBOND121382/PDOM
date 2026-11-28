"""Matched-seed strong PBDOO radius sensitivity; production algorithms unchanged."""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np
from adv_setting import loop_PBDOO_LR_strong as ps
from optimization_utils.BanditLogisticRegression import _load_bodyfat_dataset, _compute_bounds
from final_tuned_support import write_csv

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/adv_setting/bodyfat'
R = 10.
MU = 1.
TARGET = 'PBDOO-strong'
VERSION = 'strong-pbdoo-radius-diagnostics-v1'
BASE = (0.06,0.08,0.12)
HORIZONS = (100000,300000,1000000)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True),encoding='utf-8')
    tmp.replace(path)

def config(r):
    return (int(r['T']),int(r['num_nodes']),r['topology'],float(r['epsilon']))

def params(X,y,T,n,top,eps,mult):
    d = X.shape[1]
    _, M = _compute_bounds(R=R,alpha_reg=MU,max_abs_b=float(np.max(np.abs(y))))
    W = ps._build_row_stochastic_W_from_topology(top,n)
    theta = ps._estimate_theta_from_W(W,K=min(T,5000),z_floor=1e-12)
    lam,Cmix = ps._estimate_lambda_and_C(W,ps._left_perron_vector_row_stochastic(W),K=300)
    L = ps._estimate_subgradient_bound_L_for_ridge_ls(X,y,R,MU)
    U = ps._compute_Upsilon_paper(n=n,d=d,m=d,mu=MU,theta=theta,Cmix=Cmix,lam=lam,M=M,eps=eps)
    raw = ps._compute_delta_star_paper(T=T,n=n,L=L,Upsilon=U)
    delta = min(mult*raw,0.9*R)
    sigma1 = 2*d**1.5*M/(MU*eps*delta)
    assert 0 < delta <= 0.9*R and all(math.isfinite(v) for v in (raw,delta,sigma1,theta))
    return dict(delta_star=raw,delta_used=delta,cap_active=mult*raw>0.9*R,step_at_1=1/(MU*theta),noise_scale_at_1=sigma1)

def run(task):
    k,r,p = task
    X,y = _load_bodyfat_dataset(DATA)
    T,n,top,eps = config(r)
    start = time.monotonic()
    with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
        _,_,mse,loss,regret = ps.run_pbddo_linear_regression_strong(X,y,T=T,num_nodes=n,epsilon=eps,
            R=R,alpha_reg=MU,network_topology=top,rng=np.random.default_rng(int(r['seed'])),delta=p['delta_used'])
    assert all(math.isfinite(v) for v in (mse,loss,regret))
    return k,dict(avg_regret_per_round=float(regret),avg_loss_per_round=float(loss),mse=float(mse),
                  normalized_regret=float(regret)/n,elapsed_seconds=time.monotonic()-start)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',choices=['epsilon','topology'],default='epsilon')
    p.add_argument('--multipliers',nargs='+',type=float,choices=BASE,default=list(BASE))
    p.add_argument('--archived',type=Path,default=ROOT/'replay_convex_pbdoo_c'/'complete')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--validate-only',action='store_true')
    a=p.parse_args()
    assert a.workers>0 and len(set(a.multipliers))==len(a.multipliers)
    out=a.output.resolve()
    src=a.archived.resolve()
    if out==src or src in out.parents or out in src.parents:
        raise ValueError('Use separate input and output directories')
    out.mkdir(parents=True,exist_ok=True)
    exp='impact_epsilon_and_T' if a.scope=='epsilon' else 'impact_topology'
    archive=src/f'{exp}_alpha1_runs.csv'
    with archive.open(newline='') as h:
        rows=list(csv.DictReader(h))
    if len(rows)!=(700 if a.scope=='epsilon' else 840):
        raise ValueError('Wrong archive row count')
    old=[r for r in rows if r['algorithm']==TARGET]
    if len(old)!=210 or any(r['parameter_version']!='final-tuned-v1' or float(r['alpha_reg'])!=1 for r in old):
        raise ValueError('Wrong strong PBDOO archive')
    target=[r for r in old if int(r['T']) in HORIZONS and int(r['run']) in (0,1,2)]
    assert len(target)==27 and len({(config(r),r['run']) for r in target})==27
    X,y=_load_bodyfat_dataset(DATA)
    paramrows=[]; tasks={}
    for cfg in sorted({config(r) for r in target}):
        T,n,top,eps=cfg
        for mult in a.multipliers:
            q=params(X,y,*cfg,mult)
            paramrows.append(dict(T=T,num_nodes=n,topology=top,epsilon=eps,multiplier=mult,**q))
            for r in target:
                if config(r)==cfg:
                    k=f'{T}|{n}|{top}|{eps}|{mult}|{r["run"]}|{r["seed"]}'
                    tasks[k]=(k,r,q,mult)
    assert len(tasks)==27*len(a.multipliers)
    # Hash the actual inputs and implementation used by the diagnostic.
    code=[Path(__file__).resolve(),ROOT/'adv_setting/loop_PBDOO_LR_strong.py',ROOT/'optimization_utils/BanditLogisticRegression.py',ROOT/'optimization_utils/adv_setting_utils.py',ROOT/'final_tuned_support.py']
    manifest=dict(version=VERSION,archive_sha256=digest(archive),dataset_sha256=digest(DATA),
        implementation={str(f.relative_to(ROOT)):digest(f) for f in code},
        scope=a.scope,multipliers=a.multipliers,count=len(tasks),python=sys.version,numpy=np.__version__)
    mp=out/'manifest.json'; cp=out/'checkpoint.json'
    if mp.exists():
        if json.loads(mp.read_text())!=manifest:
            raise RuntimeError('Input/code/environment changed; use fresh output')
    else:
        if cp.exists():
            raise RuntimeError('Checkpoint missing manifest')
        atomic_json(mp,manifest)
    done=json.loads(cp.read_text()) if cp.exists() else {}
    if not set(done)<=set(tasks):
        raise ValueError('Unknown checkpoint keys')
    for v in done.values():
        if not all(math.isfinite(float(v[m])) for m in ('avg_regret_per_round','avg_loss_per_round','mse','normalized_regret')):
            raise ValueError('Invalid checkpoint metric')
    write_csv(out/'parameter_validation.csv',paramrows)
    print(f'Validated {len(tasks)} tasks; completed {len(done)}',flush=True)
    if a.validate_only:
        return
    # Same seed and original multiplier: three short archived checks, never mixed into candidate results.
    audit=out/'archive_reproduction.csv'
    if not audit.exists():
        checks=[]
        groups={}
        for r in old:
            if int(r['T'])==1000 and int(r['run'])==0:
                groups[config(r)]=r
        assert len(groups)==3
        for cfg,r in sorted(groups.items()):
            _,v=run(('audit',r,params(X,y,*cfg,0.04)))
            ref=float(r['avg_regret_per_round'])
            ok=math.isclose(ref,v['avg_regret_per_round'],rel_tol=1e-8,abs_tol=1e-8)
            checks.append(dict(T=cfg[0],num_nodes=cfg[1],topology=cfg[2],epsilon=cfg[3],
                seed=r['seed'],archived_regret=ref,reproduced_regret=v['avg_regret_per_round'],passed=ok))
        write_csv(audit,checks)
    with audit.open(newline='') as h:
        checks=list(csv.DictReader(h))
    if len(checks)!=3 or any(r['passed']!='True' for r in checks):
        raise RuntimeError('Archived reproduction mismatch; investigate without changing parameters')
    pending=sorted((v for k,v in tasks.items() if k not in done),key=lambda t:int(t[1]['T'])*int(t[1]['num_nodes']),reverse=True)
    if pending:
        queue=iter(pending)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            futures={}
            def submit():
                t=next(queue,None)
                if t is not None:
                    futures[pool.submit(run,t[:3])]=t[0]
            for _ in range(a.workers*2):
                submit()
            while futures:
                ready,_=wait(futures,return_when=FIRST_COMPLETED)
                for future in ready:
                    key=futures.pop(future)
                    k,v=future.result()
                    if k!=key or k in done:
                        raise RuntimeError('Invalid task key')
                    done[k]=v
                    atomic_json(cp,done)
                    print(f'{len(done)}/{len(tasks)} {k}',flush=True)
                    submit()
    assert len(done)==len(tasks)
    if digest(archive)!=manifest['archive_sha256'] or digest(DATA)!=manifest['dataset_sha256'] or any(digest(f)!=manifest['implementation'][str(f.relative_to(ROOT))] for f in code):
        raise RuntimeError('Input/code changed during execution')
    records=[]
    for k,r,q,mult in sorted(tasks.values(),key=lambda t:(t[3],config(t[1]),int(t[1]['run']))):
        rec={i:r[i] for i in ('experiment','algorithm','alpha_reg','T','num_nodes','topology','epsilon','run','seed')}
        rec.update(multiplier=mult,**q,**done[k],archived_normalized_regret=float(r['avg_regret_per_round'])/int(r['num_nodes']))
        records.append(rec)
    write_csv(out/'runs.csv',records)
    summary=[]
    for q in paramrows:
        group=[r for r in records if float(r['multiplier'])==q['multiplier'] and config(r)==(q['T'],q['num_nodes'],q['topology'],q['epsilon'])]
        assert len(group)==3
        v=[r['normalized_regret'] for r in group]
        old=[r['archived_normalized_regret'] for r in group]
        summary.append(dict(**q,num_runs=3,normalized_regret_mean=statistics.mean(v),
            normalized_regret_sd=statistics.stdev(v),archive_mean=statistics.mean(old),
            archive_sd=statistics.stdev(old)))
    write_csv(out/'summary.csv',summary)
    print(f'Finished {len(records)} candidate runs, 3 reproduction checks; no automatic selection.',flush=True)

if __name__=='__main__':
    main()
