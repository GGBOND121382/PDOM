"""Parameter preflight, conservative automatic parallelism, and final comparisons."""
import contextlib
import csv
import io
import json
import math
import os
import statistics
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from optimization_utils.BanditLogisticRegression import _build_network_topology, _build_accelerated_gossip_matrix, _compute_bounds
from adv_setting.loop_BDOO_LR import _compute_xi
from adv_setting import loop_PBDOO_LR as pc, loop_PBDOO_LR_strong as ps
from adv_setting import loop_PTF_PBDOO_LR as fc, loop_PTF_PDOCO_LR_strong as fs


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Empty output: ' + str(path))
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def validate_parameters(rows, X, y, path):
    configs = sorted({(int(float(r['alpha_reg'])), r['algorithm'], int(r['num_nodes']), r['topology'], int(r['T']), float(r['epsilon']) if r['epsilon'] else 0.) for r in rows})
    result = []
    d, R, r = X.shape[1], 10., 10.
    for a, alg, n, top, T, eps in configs:
        L, M = _compute_bounds(R=R, alpha_reg=a, max_abs_b=float(np.max(np.abs(y))))
        v = dict(alpha_reg=a, algorithm=alg, num_nodes=n, topology=top, T=T, epsilon=eps or '', C_tau='', tau_expr='', tau='', xi_theory='', xi='', eta='', step_at_1='', noise_scale_at_1='', num_blocks='', theta='', lambda_mix='', Cmix='')
        if alg == 'BDOO':
            _, _, rho, _ = _build_network_topology(top, n)
            if not 0 < rho < 1:
                raise ValueError('Invalid BDOO rho: ' + str(v))
            with contextlib.redirect_stdout(io.StringIO()):
                raw = r*_compute_xi(setting='strongly_convex' if a else 'convex', T=T, dim=d, num_nodes=n, R=R, r=r, L_f=L, C=M, rho=rho, alpha=a or None)
            xi = min((0.2 if a else 0.02)*raw, 0.9*r)
            v.update(xi_theory=raw, xi=xi, step_at_1=1/a if a else 4*xi*R/(d*M))
        elif alg == 'PBDOO-cor1':
            A = pc._build_pushsum_matrix_from_topology(top, n)
            L = pc.lipschitz_bound_squared_loss(X, y, R, a)
            raw = pc.delta_star_corollary1_fixed_point(n=n, d=d, T=T, C=M, L=L, zeta=3e-4, epsilon=eps, A=A, l_i=0., delta_init=0.1*R, iters=50)
            xi = min(0.3*raw, 0.9*R)
            v.update(xi_theory=raw, xi=xi, step_at_1=3e-4, noise_scale_at_1=2*d**1.5*M/(eps*xi))
        elif alg == 'PBDOO-strong':
            W = ps._build_row_stochastic_W_from_topology(top, n)
            theta = ps._estimate_theta_from_W(W, K=min(T,5000), z_floor=1e-12)
            lam, mix = ps._estimate_lambda_and_C(W, ps._left_perron_vector_row_stochastic(W), K=300)
            L = ps._estimate_subgradient_bound_L_for_ridge_ls(X,y,R,a)
            U = ps._compute_Upsilon_paper(n=n,d=d,m=d,mu=a,theta=theta,Cmix=mix,lam=lam,M=M,eps=eps)
            raw = ps._compute_delta_star_paper(T=T,n=n,L=L,Upsilon=U)
            xi = min(0.04*raw, 0.9*R)
            v.update(xi_theory=raw, xi=xi, step_at_1=1/(a*theta), noise_scale_at_1=2*d**1.5*M/(a*eps*xi), theta=theta, lambda_mix=lam, Cmix=mix)
        elif alg.startswith('PTF-'):
            P = _build_accelerated_gossip_matrix(top,n)
            kw = dict(T=T,d=d,n=n,epsilon=eps,r=r,C_norm=R,P=P,L=L,M=M)
            fixed = 'fixed-block' in alg
            if a:
                kw['alpha'] = a
                tau, xi, floor = (fs.final_tuned_fixed_block_strong if fixed else fs.final_tuned_corollary2_params)(**kw)
                expr = (a*a*d**4*M*M*(M+L*r)**2*T*T/(n*r*r*eps*eps*(L*L+a*a*R*R)**3*math.log(T)**2))**0.25
                raw = max((d*d*M*M*r*math.log(T)/(a*(M+L*r)*T)*(1+d*d/(n*tau*eps*eps)))**(1/3), (L*d*d*M*r*math.log(T)/(a*eps*(M+L*r)*T))**0.5)
                assert math.isclose(xi,min(0.75*raw,0.9*r),rel_tol=1e-12)
                eta = ''
            else:
                tau, xi, eta, floor = (fc.final_tuned_fixed_block_convex if fixed else fc.final_tuned_corollary1_params)(**kw)
                expr = (d**4*M*M*(M+L*r)**2*T/(n*L**4*r*r*R*R*eps*eps))**(1/3)
                raw = max((d*d*M*M*r*r*R*R/(T*(M+L*r)**2)*(1+d*d/(n*tau*eps*eps)))**0.25, (L*d*d*M*r*r*R*R/(eps*T*(M+L*r)**2))**(1/3))
                assert math.isclose(xi,min(raw,0.9*r),rel_tol=1e-12)
                assert math.isclose(eta,1.5*R/math.sqrt(T*(d*d*M*M/xi**2*(1+d*d/(n*tau*eps*eps))+tau*L*L+L*d*d*M/(eps*xi))),rel_tol=1e-12)
            expected_tau = math.ceil(floor if fixed else max(floor,(6 if a else 1.4)*expr))
            if not floor <= tau <= T or tau != expected_tau:
                raise ValueError('Invalid tau: ' + str((v,tau,floor,expected_tau)))
            v.update(C_tau=floor,tau_expr=expr,tau=tau,xi_theory=raw,xi=xi,eta=eta,noise_scale_at_1=2*d**1.5*M/(eps*xi),num_blocks=math.ceil(T/tau))
        else:
            raise ValueError('Unknown algorithm: ' + alg)
        if not 0 < v['xi'] <= 0.9*r or any(not math.isfinite(x) for x in v.values() if isinstance(x,(float,int))):
            raise ValueError('Invalid parameters: ' + str(v))
        result.append(v)
    write_csv(path,result)


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ''


def resource_snapshot():
    cpu = float(len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else (os.cpu_count() or 1))
    try:
        import psutil
        available = psutil.virtual_memory().available
    except ImportError:
        available = None
        for line in _read('/proc/meminfo').splitlines():
            if line.startswith('MemAvailable:'):
                available = int(line.split()[1])*1024
                break
    # Check ancestor limits in cgroup v2 (including containers with a CPU quota).
    cgroups = {Path('/sys/fs/cgroup')}
    for line in _read('/proc/self/cgroup').splitlines():
        if line.startswith('0::'):
            q = Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')
            while q == Path('/sys/fs/cgroup') or Path('/sys/fs/cgroup') in q.parents:
                cgroups.add(q)
                q = q.parent
    for q in cgroups:
        quota = _read(q/'cpu.max').split()
        if len(quota)==2 and quota[0]!='max':
            cpu = min(cpu, int(quota[0])/int(quota[1]))
        limit, used = _read(q/'memory.max'), _read(q/'memory.current')
        if limit.isdigit() and used.isdigit():
            free = max(0,int(limit)-int(used))
            available = free if available is None else min(available,free)
    # Legacy cgroup v1 fallback.
    quota, period = _read('/sys/fs/cgroup/cpu/cpu.cfs_quota_us'), _read('/sys/fs/cgroup/cpu/cpu.cfs_period_us')
    if quota.isdigit() and period.isdigit() and int(period)>0:
        cpu = min(cpu,int(quota)/int(period))
    limit, used = _read('/sys/fs/cgroup/memory/memory.limit_in_bytes'), _read('/sys/fs/cgroup/memory/memory.usage_in_bytes')
    if limit.isdigit() and used.isdigit() and int(limit)<2**60:
        free = max(0,int(limit)-int(used))
        available = free if available is None else min(available,free)
    return dict(cpu_effective=cpu, available_bytes=available, logical_cpu=os.cpu_count())


def auto_workers(out,pending,runner,save,*,pilot,bounded):
    info = resource_snapshot()
    # Streaming implementations keep sample indices and a small number of arrays.
    # Allow four full T*n arrays plus 512 MiB for interpreter/BLAS/other buffers.
    peak = 512*1024**2 + 32*max(int(r['T'])*int(r['num_nodes']) for _,r in pending)
    # Pilot must not estimate memory using only short horizons.
    peak = max(peak,512*1024**2+32*1000000*36)
    free = info['available_bytes']
    cap = max(1,min(len(pending),math.floor(info['cpu_effective']), math.floor(0.8*free/peak) if free is not None else 1))
    report = dict(mode='auto', resources=info, estimated_peak_bytes=peak, cap=cap, selected=cap, benchmark=[])
    # Benchmark exactly the same small balanced batch for every candidate.
    # In --max-runs and pilot mode, avoid extra repeated tasks.
    if not pilot and not bounded and cap>1:
        groups=defaultdict(list)
        for item in pending:
            k,r=item
            if int(r['T'])<=10000:
                groups[(r['alpha_reg'],r['algorithm'],r['topology'])].append(item)
        batch=[]
        while len(batch)<min(64,max(16,2*cap)) and any(groups.values()):
            for key in sorted(groups):
                if groups[key] and len(batch)<min(64,max(16,2*cap)):
                    batch.append(groups[key].pop())
        if len(batch)>=4:
            candidates=sorted({max(1,math.ceil(cap*0.5)),max(1,math.ceil(cap*0.75)),cap})
            for w in candidates:
                start=time.monotonic(); rss=0
                with ProcessPoolExecutor(max_workers=w) as pool:
                    for fut in as_completed([pool.submit(runner,t) for t in batch]):
                        k,v,elapsed,memory=fut.result();save(k,v);rss=max(rss,memory)
                wall=time.monotonic()-start
                report['benchmark'].append(dict(workers=w,seconds=wall,tasks=len(batch),tasks_per_second=len(batch)/wall,peak_worker_rss=rss))
            report['selected']=max(report['benchmark'],key=lambda b:b['tasks_per_second'])['workers']
    (out/'resource_benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report['selected']


def compare_results(old,new,out,files):
    out.mkdir(exist_ok=True)
    history=[]; cells=defaultdict(lambda:defaultdict(list))
    for f in files:
        def grouped(root):
            g=defaultdict(list)
            with (root/f).open(newline='') as h:
                for r in csv.DictReader(h):
                    k=(int(float(r['alpha_reg'])),int(r['num_nodes']),r['topology'],int(r['T']),float(r['epsilon']) if r['epsilon'] else 0.,r['algorithm'])
                    g[k].append(float(r['avg_regret_per_round'])/k[1])
            return g
        before,after=grouped(old),grouped(new)
        if before.keys()!=after.keys():
            raise ValueError('Comparison configuration mismatch: '+f)
        for k,v in after.items():
            x,y=statistics.mean(before[k]),statistics.mean(v)
            history.append(dict(file=f,alpha=k[0],n=k[1],topology=k[2],T=k[3],epsilon=k[4] or '',algorithm=k[5],old_mean=x,new_mean=y,new_sd=statistics.stdev(v),difference=y-x))
            cells[k[:-1]][k[-1]].append(y)
    write_csv(out/'historical_comparison.csv',history)
    result=[]
    for key,algs in sorted(cells.items()):
        a,n,top,T,eps=key
        if not eps:
            continue
        ours=statistics.mean(algs['PTF-PBDOO-strong' if a else 'PTF-PBDOO-cor1'])
        pb=statistics.mean(algs['PBDOO-strong' if a else 'PBDOO-cor1'])
        van=statistics.mean(algs['PTF-PBDOO-fixed-block-strong' if a else 'PTF-PBDOO-fixed-block-optimized-eta-cor1'])
        bd=statistics.mean(cells[(a,n,top,T,0.)]['BDOO'])
        result.append(dict(alpha=a,n=n,topology=top,T=T,epsilon=eps,ours=ours,pbdoo=pb,bdoo=bd,vanilla=van,pbdoo_reduction=1-ours/pb if pb>1e-12 else '',bdoo_ratio=ours/bd if bd>1e-12 else '',bdoo_difference=ours-bd,vanilla_reduction=1-ours/van if van>1e-12 else ''))
    write_csv(out/'per_config.csv',result)
    report=[]
    for a in (0,1):
        r=[v for v in result if v['alpha']==a and v['T']>=100000]
        p=[v['pbdoo_reduction'] for v in r if v['pbdoo_reduction']!='']
        report.append(dict(alpha=a,settings=len(r),valid_ratios=len(p),mean_pbdoo_reduction=statistics.mean(p) if p else '',fraction_at_least_50pct=sum(x>=.5 for x in p)/len(p) if p else ''))
    write_csv(out/'long_horizon_summary.csv',report)
    print('Long-horizon comparison:',report,flush=True)
