"""Matched-seed convex Vanilla-block sensitivity; production algorithms unchanged."""
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
from adv_setting import loop_PTF_PBDOO_LR as pc
from optimization_utils.BanditLogisticRegression import _load_bodyfat_dataset, _compute_bounds, _make_loss_oracle, _build_accelerated_gossip_matrix
from optimization_utils.adv_setting_utils import StreamingAvgLossRegretAccumulator
from final_tuned_support import write_csv

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'data/adv_setting/bodyfat'
R = 10.
ALPHA = 0.
TARGET = 'PTF-PBDOO-fixed-block-optimized-eta-cor1'
VERSION = 'convex-vanilla-block-diagnostics-v1'
BASE = {'A': (1.,1.), 'B': (1.25,1.5), 'C': (1.25,1.)}
HORIZONS = (100000,300000,1000000)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic_json(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True),encoding='utf-8')
    tmp.replace(path)

def config(r):
    return (int(r['T']),int(r['num_nodes']),r['topology'],float(r['epsilon']))

def params(X,y,T,n,top,eps,candidate):
    d = X.shape[1]
    c_xi,c_eta = (1.,1.5) if candidate=='archive' else BASE[candidate]
    L,M = pc._parse_compute_bounds_output(_compute_bounds(R=R,alpha_reg=ALPHA,max_abs_b=float(np.max(np.abs(y)))))
    P = _build_accelerated_gossip_matrix(top,n)
    tau,C_tau = pc._fixed_block_size(T=T,d=d,n=n,epsilon=eps,r=R,C_norm=R,P=P,L=L,M=M)
    factor = 1.+d**2/(n*tau*eps**2)
    xi_bandit = (d**2*M**2*R**4/(T*(M+L*R)**2)*factor)**0.25
    xi_privacy = (L*d**2*M*R**4/(eps*T*(M+L*R)**2))**(1./3.)
    raw = max(xi_bandit,xi_privacy)
    xi = min(c_xi*raw,0.9*R)
    eta = c_eta*R/math.sqrt(T*((d**2*M**2/xi**2)*factor+tau*L**2+L*d**2*M/(eps*xi)))
    sigma = 2*d**1.5*M/(eps*xi)
    assert 0<xi<=0.9*R and eta>0 and all(math.isfinite(v) for v in (raw,xi,eta,sigma))
    if c_xi==1.:
        check_tau,check_xi,check_eta,_ = pc.fixed_block_optimized_eta_params_convex(
            T=T,d=d,n=n,epsilon=eps,r=R,C_norm=R,P=P,L=L,M=M,eta_multiplier=c_eta)
        assert tau==check_tau and math.isclose(xi,check_xi,rel_tol=1e-12) and math.isclose(eta,check_eta,rel_tol=1e-12)
    return dict(xi_multiplier=c_xi,eta_multiplier=c_eta,tau=tau,C_tau=C_tau,xi_star=raw,xi_used=xi,
        cap_active=c_xi*raw>0.9*R,eta=eta,noise_scale=sigma)

def run(task):
    k,r,p = task
    X,y = _load_bodyfat_dataset(DATA)
    T,n,top,eps = config(r)
    rng = np.random.default_rng(int(r['seed']))
    samples = rng.integers(0,len(X),size=(T,n))
    loss_oracle = _make_loss_oracle(X,y,samples,ALPHA)
    def loss_fn(t,i,x):
        return float(loss_oracle(i,t,x))
    L,M = pc._parse_compute_bounds_output(_compute_bounds(R=R,alpha_reg=ALPHA,max_abs_b=float(np.max(np.abs(y)))))
    P = _build_accelerated_gossip_matrix(top,n)
    acc = StreamingAvgLossRegretAccumulator(samples=samples,X=X,y=y,alpha_reg=ALPHA,R=R)
    cfg = pc.PTF_PBDO2_Convex_Config(n=n,d=X.shape[1],T=T,tau=p['tau'],eta=p['eta'],xi=p['xi_used'],
        epsilon=eps,M=float(M),P=P,R=R,r=R,seed=0,rng=rng,store_actions=False,
        follow_paper_delay=True,action_callback=lambda t,x:acc.update(x))
    start = time.monotonic()
    with open(os.devnull,'w') as sink,contextlib.redirect_stdout(sink):
        res = pc.run_ptf_pbd02_convex(loss_fn,cfg)
    w_hat = res['x_batches'].mean(axis=(0,1))
    mse = float(np.mean((X@w_hat-y)**2))
    loss,regret = acc.finalize()
    assert all(math.isfinite(v) for v in (mse,loss,regret))
    return k,dict(avg_regret_per_round=float(regret),avg_loss_per_round=float(loss),mse=mse,
                  normalized_regret=float(regret)/n,elapsed_seconds=time.monotonic()-start)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scope',choices=['epsilon','topology'],default='epsilon')
    p.add_argument('--candidates',nargs='+',choices=list(BASE),default=list(BASE))
    p.add_argument('--archived',type=Path,default=ROOT/'replay_convex_pbdoo_c'/'complete')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--validate-only',action='store_true')
    a=p.parse_args()
    assert a.workers>0 and len(set(a.candidates))==len(a.candidates)
    out=a.output.resolve()
    src=a.archived.resolve()
    if out==src or src in out.parents or out in src.parents:
        raise ValueError('Use separate input and output directories')
    out.mkdir(parents=True,exist_ok=True)
    exp='impact_epsilon_and_T' if a.scope=='epsilon' else 'impact_topology'
    archive=src/f'{exp}_alpha0_runs.csv'
    with archive.open(newline='') as h:
        rows=list(csv.DictReader(h))
    if len(rows)!=(700 if a.scope=='epsilon' else 840):
        raise ValueError('Wrong archive row count')
    old=[r for r in rows if r['algorithm']==TARGET]
    if len(old)!=210 or any(r['parameter_version']!='final-tuned-v1' or float(r['alpha_reg'])!=0 for r in old):
        raise ValueError('Wrong convex Vanilla-block archive')
    target=[r for r in old if int(r['T']) in HORIZONS and int(r['run']) in (0,1,2)]
    assert len(target)==27 and len({(config(r),r['run']) for r in target})==27
    X,y=_load_bodyfat_dataset(DATA)
    paramrows=[]; tasks={}
    for cfg in sorted({config(r) for r in target}):
        T,n,top,eps=cfg
        for mult in a.candidates:
            q=params(X,y,*cfg,mult)
            paramrows.append(dict(T=T,num_nodes=n,topology=top,epsilon=eps,candidate=mult,**q))
            for r in target:
                if config(r)==cfg:
                    k=f'{T}|{n}|{top}|{eps}|{mult}|{r["run"]}|{r["seed"]}'
                    tasks[k]=(k,r,q,mult)
    assert len(tasks)==27*len(a.candidates)
    # Hash the actual inputs and implementation used by the diagnostic.
    code=[Path(__file__).resolve(),ROOT/'adv_setting/loop_PTF_PBDOO_LR.py',ROOT/'optimization_utils/BanditLogisticRegression.py',ROOT/'optimization_utils/adv_setting_utils.py',ROOT/'final_tuned_support.py']
    manifest=dict(version=VERSION,archive_sha256=digest(archive),dataset_sha256=digest(DATA),
        implementation={str(f.relative_to(ROOT)):digest(f) for f in code},
        scope=a.scope,candidates=a.candidates,count=len(tasks),python=sys.version,numpy=np.__version__)
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
    # Same seed and original multipliers: three short archived checks, never mixed into candidate results.
    audit=out/'archive_reproduction.csv'
    if not audit.exists():
        checks=[]
        groups={}
        for r in old:
            if int(r['T'])==1000 and int(r['run'])==0:
                groups[config(r)]=r
        assert len(groups)==3
        for cfg,r in sorted(groups.items()):
            _,v=run(('audit',r,params(X,y,*cfg,'archive')))
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
        rec.update(candidate=mult,**q,**done[k],archived_normalized_regret=float(r['avg_regret_per_round'])/int(r['num_nodes']))
        records.append(rec)
    write_csv(out/'runs.csv',records)
    summary=[]
    for q in paramrows:
        group=[r for r in records if r['candidate']==q['candidate'] and config(r)==(q['T'],q['num_nodes'],q['topology'],q['epsilon'])]
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
