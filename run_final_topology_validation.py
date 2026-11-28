#!/usr/bin/env python3
"""Run and audit the two pinned, matched-seed topology diagnostics; never freeze parameters."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SPECS = (
    ('strong-pbdoo-radius-diagnostics', '4fbcf270f49a8cc97db944833fdc19198b02ea2b',
     'diagnose_strong_pbdoo_radius.py', '--multipliers', '0.08',
     'replay_strong_pbdoo_radius_topology', 1, 'PBDOO-strong', 'multiplier'),
    ('convex-vanilla-block-diagnostics', 'ec4fd3560c6bcc28e919f862547575aaf48a6c5b',
     'diagnose_convex_vanilla_block.py', '--candidates', 'C',
     'replay_convex_vanilla_block_topology', 0, 'PTF-PBDOO-fixed-block-optimized-eta-cor1', 'candidate'),
)
TOPS = ('cycle', 'grid', 'complete')
TS = (100000, 300000, 1000000)


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-10, abs_tol=1e-10)


def key(r):
    return (r['topology'].lower(), int(r['T']), int(r['num_nodes']), float(r['epsilon']), int(r['run']), int(r['seed']))


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def audit(work, spec):
    branch, sha, script, option, candidate, dirname, alpha, algorithm, field = spec
    out = work / dirname
    src = work / 'replay_convex_pbdoo_c' / 'complete' / f'impact_topology_alpha{alpha}_runs.csv'
    archive = read(src)
    require(len(archive) == 840, 'Wrong topology archive size')
    target = [r for r in archive if r['algorithm'] == algorithm and int(r['T']) in TS and int(r['run']) in (0,1,2)]
    expected = {key(r): r for r in target}
    require(len(target) == len(expected) == 27, 'Wrong archive target count')
    require({k[:5] for k in expected} == {(t,T,36,1.,i) for t in TOPS for T in TS for i in range(3)}, 'Wrong topology grid')
    runs = read(out/'runs.csv')
    require(len(runs) == 27 and {key(r) for r in runs} == set(expected), 'Missing, duplicate, or altered seed/configuration')
    manifest = json.loads((out/'manifest.json').read_text())
    checkpoint = json.loads((out/'checkpoint.json').read_text())
    require(manifest['scope']=='topology' and manifest['count']==len(checkpoint)==27, 'Wrong manifest/checkpoint')
    checks = read(out/'archive_reproduction.csv')
    require(len(checks)==3 and all(r['passed']=='True' and math.isclose(float(r['archived_regret']),float(r['reproduced_regret']),rel_tol=1e-8,abs_tol=1e-8) for r in checks), 'Archive reproduction failed')
    params = read(out/'parameter_validation.csv')
    summary = read(out/'summary.csv')
    groups = {}
    for r in runs:
        k = key(r)
        require(r['algorithm']==algorithm and float(r['alpha_reg'])==alpha and r['experiment']=='impact_topology', 'Wrong algorithm or experiment')
        require(close(r[field], candidate) if alpha else r[field]==candidate, 'Wrong candidate')
        require(all(math.isfinite(float(r[m])) for m in ('avg_regret_per_round','avg_loss_per_round','mse','normalized_regret')), 'Non-finite metric')
        require(close(r['normalized_regret'],float(r['avg_regret_per_round'])/k[2]) and close(r['archived_normalized_regret'],float(expected[k]['avg_regret_per_round'])/k[2]), 'Regret normalization/archive mismatch')
        if not alpha:
            require(close(r['xi_multiplier'],1.25) and close(r['eta_multiplier'],1.) and int(r['tau'])==math.ceil(float(r['C_tau'])), 'Wrong Vanilla parameters')
        groups.setdefault(k[:4], []).append(float(r['normalized_regret']))
    def cfg(r):
        return (r['topology'].lower(),int(r['T']),int(r['num_nodes']),float(r['epsilon']))
    require(len(params)==len(summary)==9 and {cfg(r) for r in params}=={cfg(r) for r in summary}==set(groups), 'Wrong parameter/summary coverage')
    for r in summary:
        vals=groups[cfg(r)]
        require(len(vals)==int(r['num_runs'])==3 and close(r['normalized_regret_mean'],statistics.mean(vals)) and close(r['normalized_regret_sd'],statistics.stdev(vals)), 'Mean/sample SD mismatch')
    report = dict(branch=branch, source_commit=sha, algorithm=algorithm, candidate=candidate, candidate_runs=27, reproduction_checks=3, structural_audit='PASS', means={t:[statistics.mean(groups[(t,T,36,1.)]) for T in TS] for t in TOPS}, horizons=list(TS), parameter_selection='NOT PERFORMED')
    (out/'topology_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


def setup(workspace, spec):
    branch, sha, script, option, candidate, dirname, alpha, algorithm, field = spec
    work = workspace / branch
    git('fetch','origin',f'refs/heads/{branch}')
    require(git('rev-parse','FETCH_HEAD')==sha, f'{branch} advanced; inspect latest remote results before running')
    if not work.exists():
        git('worktree','add','--detach',str(work),sha)
    require(git('rev-parse','HEAD',cwd=work)==sha and not git('status','--porcelain','--untracked-files=no',cwd=work), f'{work} must be at pinned commit with no tracked changes')
    return work


def execute(work, spec, workers, preflight=False):
    branch, sha, script, option, candidate, dirname, alpha, algorithm, field = spec
    out=work/dirname
    out.mkdir(exist_ok=True)
    cmd=[sys.executable,'-u',script,'--scope','topology',option,candidate,'--output',dirname,'--workers',str(workers)]
    if preflight:
        cmd.append('--validate-only')
    print(f'[{branch}] {" ".join(cmd)}',flush=True)
    with (out/'execution.log').open('a',encoding='utf-8') as log:
        with subprocess.Popen(cmd,cwd=work,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as proc:
            for line in proc.stdout:
                print(line,end='',flush=True)
                log.write(line)
                log.flush()
            require(proc.wait()==0,f'{branch} failed; inspect execution.log, retain checkpoint')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=ROOT.parent/(ROOT.name+'_topology_validation'))
    p.add_argument('--workers',type=int,choices=range(1,5),default=4)
    p.add_argument('--validate-only',action='store_true')
    a=p.parse_args()
    workspace=a.workspace.resolve()
    require(workspace!=ROOT and ROOT not in workspace.parents,'Workspace must be outside the source checkout')
    workspace.mkdir(parents=True,exist_ok=True)
    jobs=[(setup(workspace,s),s) for s in SPECS]
    for work,s in jobs:
        execute(work,s,a.workers,preflight=True)
    if a.validate_only:
        print('Both preflights passed; no candidate runs executed.')
        return
    for work,s in jobs:
        execute(work,s,a.workers)
        audit(work,s)
    print('54 candidate runs + 6 reproduction checks audited. No parameter selection or formal replay performed.')


if __name__=='__main__':
    main()
