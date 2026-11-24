"""Replay proposed and Vanilla-block algorithms with lazy Metropolis weights."""
import argparse,csv,hashlib,json,math,os,pathlib,contextlib,io,subprocess,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from adv_setting.adv_setting_common import ALGORITHMS_ALPHA0,ALGORITHMS_ALPHA1,FIXED_BLOCK_OPTIMIZED_ETA_ALPHA0,FIXED_BLOCK_ALPHA1,validate_run_rows
from optimization_utils.BanditLogisticRegression import _load_bodyfat_dataset
ROOT=pathlib.Path(__file__).resolve().parent
SPECS={(int(s.alpha_reg),s.name):s for s in ALGORITHMS_ALPHA0+ALGORITHMS_ALPHA1+[FIXED_BLOCK_OPTIMIZED_ETA_ALPHA0,FIXED_BLOCK_ALPHA1]}
FILES=[f'{e}_alpha{a}_runs.csv' for e in ('impact_epsilon_and_T','impact_topology') for a in (0,1)]
AFFECTED={'PTF-PBDOO-cor1','PTF-PBDOO-strong','PTF-PBDOO-fixed-block-optimized-eta-cor1','PTF-PBDOO-fixed-block-strong'}
METRICS=('avg_regret_per_round','avg_loss_per_round','mse')
def digest(p):
 b=p.read_bytes()
 return hashlib.sha256(b.replace(b'\r\n',b'\n') if p.suffix=='.py' else b).hexdigest()
def atom_json(p,obj):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,sort_keys=True,indent=2));t.replace(p)
def run(task):
 k,r=task;s=SPECS[int(float(r['alpha_reg'])),r['algorithm']];X,y=_load_bodyfat_dataset(ROOT/'data/adv_setting/bodyfat')
 kw=dict(T=int(r['T']),num_nodes=int(r['num_nodes']),topology=r['topology'],rng=np.random.default_rng(int(r['seed'])))
 if s.uses_epsilon:kw['epsilon']=float(r['epsilon'])
 with contextlib.redirect_stdout(io.StringIO()):v=s.runner(X,y,**kw)
 if not all(math.isfinite(x) for x in v):raise RuntimeError('nonfinite result '+k)
 return k,dict(zip(METRICS,map(float,v)))
def check_result(k,v):
 if set(v) not in (set(METRICS),{METRICS[0]}) or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in v.values()):raise ValueError('invalid checkpoint metrics '+k)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archived',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);p.add_argument('--workers',type=int,default=1);p.add_argument('--max-runs',type=int);p.add_argument('--pilot',action='store_true',help='Only 16 short-horizon runs, retained for full replay');a=p.parse_args();src=a.archived.resolve();out=a.output.resolve()
 if src==out or src in out.parents or out in src.parents:raise ValueError('source/output must be separate')
 if a.workers<1 or (a.max_runs is not None and a.max_runs<0):raise ValueError('invalid run limit')
 out.mkdir(parents=True,exist_ok=True);rows={};tasks={};reused={};fields={}
 archive_hashes={f:digest(src/f) for f in FILES}
 os.environ['PDOM_R15_TUNING']='refined'
 for f,count in zip(FILES,(700,700,840,840)):
  with (src/f).open(newline='') as h:
   reader=csv.DictReader(h);fields[f]=list(reader.fieldnames);rows[f]=list(reader)
  if 'implementation' not in fields[f]:fields[f].append('implementation')
  if 'gossip_matrix' not in fields[f]:fields[f].append('gossip_matrix')
  if len(rows[f])!=count:raise ValueError('wrong row count '+f)
  errors=validate_run_rows(rows[f],expected_runs=10,base_seed=20261109)
  if errors:raise ValueError(f'{f}: {errors[:5]}')
  for i,r in enumerate(rows[f]):
   k=f+':'+str(i)
   if (int(float(r['alpha_reg'])),r['algorithm']) not in SPECS:raise ValueError('unexpected algorithm '+k)
   if r['algorithm'] not in AFFECTED and r.get('implementation') not in ('query-feasible-cap09-v2','query-feasible-cap09-v2-regret-only'):
    raise ValueError('Expected completed cap09 replay, not pre-cap09 archive: '+k)
   (tasks if r['algorithm'] in AFFECTED else reused)[k]=r
 if len(tasks)!=1680 or len(reused)!=1400:raise ValueError('unexpected replay scope')
 manifest={'matrix':'lazy-metropolis-v1','archived':archive_hashes,'implementation':{str(f.relative_to(ROOT)).replace('\\','/'):digest(f) for f in sorted([*ROOT.glob('adv_setting/*.py'),*ROOT.glob('optimization_utils/*.py'),ROOT/'data/adv_setting/bodyfat',pathlib.Path(__file__).resolve()])},'numpy':np.__version__}
 mp=out/'manifest.json';cp=out/'checkpoint.json'
 if mp.exists():
  if json.loads(mp.read_text())!=manifest:raise RuntimeError('source/code/environment changed; use fresh output directory')
 else:
  if cp.exists():raise RuntimeError('checkpoint lacks provenance')
  atom_json(mp,manifest)
 done=json.loads(cp.read_text()) if cp.exists() else {}
 if not set(done)<=set(tasks):raise ValueError('unknown checkpoint keys')
 for k,v in done.items():
  check_result(k,v)
  if set(v)!=set(METRICS):raise ValueError('new replay must contain all metrics: '+k)
 print(f'Archived reuse={len(reused)}, completed={len(done)}, remaining={len(tasks)-len(done)}',flush=True)
 pending=[(k,r) for k,r in tasks.items() if k not in done];pending.sort(key=lambda kv:(int(kv[1]['T']),int(kv[1]['num_nodes']),int(kv[1]['run']),kv[0]))
 if a.pilot:
  selected=[(k,r) for k,r in tasks.items() if int(r['T'])==1000 and int(r['run'])==0 and float(r['epsilon'])==1.0]
  if len(selected)!=16:raise ValueError('expected 16 pilot runs')
  pilot_keys={k for k,r in selected}
  pending=[t for t in pending if t[0] in pilot_keys]
 if a.max_runs is not None:pending=pending[:a.max_runs]
 if pending:
  with ProcessPoolExecutor(max_workers=a.workers) as pool:
   futures=[pool.submit(run,t) for t in pending]
   for future in as_completed(futures):
    k,v=future.result();done[k]=v;atom_json(cp,done);print(f'{len(done)}/{len(tasks)} {k}',flush=True)
 if len(done)!=len(tasks):
  print('Checkpoint only; no final CSV or figures published.',flush=True);return
 if {f:digest(src/f) for f in FILES}!=archive_hashes:raise RuntimeError('source changed during replay')
 dest=out/'complete';dest.mkdir(exist_ok=True)
 for f in FILES:
  temp=dest/(f+'.tmp')
  with temp.open('w',newline='') as h:
   w=csv.DictWriter(h,fieldnames=fields[f]);w.writeheader()
   for i,r in enumerate(rows[f]):
    k=f+':'+str(i);s=dict(r)
    if k in done:
     v=done[k];s.update({m:v[m] for m in METRICS})
     if int(float(r['alpha_reg']))==0:s['implementation']='paper-alg1-v1'
     else:s['implementation']='lazy-metropolis-v1'
    if k in done:s['gossip_matrix']='lazy-metropolis-v1'
    w.writerow(s)
  temp.replace(dest/f)
 subprocess.run([sys.executable,'-m','adv_setting.plot_adv_setting_results','--output_dir',str(dest),'--recompute_summary','--require_runs','10'],cwd=ROOT,check=True)
 subprocess.run([sys.executable,str(ROOT/'compare_gossip_results.py'),'--old',str(src),'--new',str(dest),'--output',str(out/'comparison')],cwd=ROOT,check=True)
 print('Complete candidate runs and figures:',dest)
if __name__=='__main__':main()
