"""Replay only affected cap09 baselines, preserving archived and pilot regret."""
import argparse,csv,hashlib,json,math,os,pathlib,contextlib,io,subprocess,sys
from concurrent.futures import ProcessPoolExecutor,as_completed
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from adv_setting.adv_setting_common import ALGORITHMS_ALPHA0,ALGORITHMS_ALPHA1,FIXED_BLOCK_OPTIMIZED_ETA_ALPHA0,FIXED_BLOCK_ALPHA1,validate_run_rows
from optimization_utils.BanditLogisticRegression import _load_bodyfat_dataset
ROOT=pathlib.Path(__file__).resolve().parent
SPECS={(int(s.alpha_reg),s.name):s for s in ALGORITHMS_ALPHA0+ALGORITHMS_ALPHA1+[FIXED_BLOCK_OPTIMIZED_ETA_ALPHA0,FIXED_BLOCK_ALPHA1]}
FILES=[f'{e}_alpha{a}_runs.csv' for e in ('impact_epsilon_and_T','impact_topology') for a in (0,1)]
AFFECTED={'BDOO','PBDOO-cor1','PBDOO-strong'}
METRICS=('avg_regret_per_round','avg_loss_per_round','mse')
ARCHIVE_HASHES=dict(zip(FILES,('daed81c8dddaae50b838b8c3ff750f9c916cf302b502b3c68e97eb88659e7b45','5b4a457cc54719e26786c9ca3fa75dbbf206a059184622b73504e71761b2eeaf','00281da2b2fa4ba552f28ae8044ef2a22eaaf4f3fa412a039d454cfeb67f3b23','81535c57c4e2ec5a8dd5a5b4799ae1552d2180efcc058aaafa636bbe51c4d760')))
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
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archived',required=True,type=pathlib.Path);p.add_argument('--output',required=True,type=pathlib.Path);p.add_argument('--workers',type=int,default=1);p.add_argument('--max-runs',type=int);p.add_argument('--verify-reused',action='store_true');p.add_argument('--seed-checkpoint',type=pathlib.Path,help='Optional audited cap09 pilot checkpoint with matching manifest');a=p.parse_args();src=a.archived.resolve();out=a.output.resolve()
 if src==out or src in out.parents or out in src.parents:raise ValueError('source/output must be separate')
 if a.workers<1 or (a.max_runs is not None and a.max_runs<0):raise ValueError('invalid run limit')
 out.mkdir(parents=True,exist_ok=True);rows={};tasks={};reused={};fields={}
 if {f:digest(src/f) for f in FILES}!=ARCHIVE_HASHES:raise RuntimeError('wrong archived CSVs; use the exact 3080-run input bundle')
 for f,count in zip(FILES,(700,700,840,840)):
  with (src/f).open(newline='') as h:
   reader=csv.DictReader(h);fields[f]=list(reader.fieldnames);rows[f]=list(reader)
  if 'implementation' not in fields[f]:fields[f].append('implementation')
  if len(rows[f])!=count:raise ValueError('wrong row count '+f)
  errors=validate_run_rows(rows[f],expected_runs=10)
  if errors:raise ValueError(f'{f}: {errors[:5]}')
  for i,r in enumerate(rows[f]):
   k=f+':'+str(i)
   if (int(float(r['alpha_reg'])),r['algorithm']) not in SPECS:raise ValueError('unexpected algorithm '+k)
   (tasks if r['algorithm'] in AFFECTED else reused)[k]=r
 if len(tasks)!=1400 or len(reused)!=1680:raise ValueError('unexpected replay scope')
 manifest={'archived':ARCHIVE_HASHES,'implementation':{str(f.relative_to(ROOT)).replace('\\','/'):digest(f) for f in sorted([*ROOT.glob('adv_setting/*.py'),*ROOT.glob('optimization_utils/*.py'),ROOT/'data/adv_setting/bodyfat',pathlib.Path(__file__).resolve()])},'numpy':np.__version__}
 mp=out/'manifest.json';cp=out/'checkpoint.json'
 if mp.exists():
  if json.loads(mp.read_text())!=manifest:raise RuntimeError('source/code/environment changed; use fresh output directory')
 else:
  if cp.exists():raise RuntimeError('checkpoint lacks provenance')
  atom_json(mp,manifest)
 done=json.loads(cp.read_text()) if cp.exists() else {}
 if a.seed_checkpoint:
  seed=json.loads(a.seed_checkpoint.read_text())
  if seed['manifest']!=manifest:raise RuntimeError('pilot manifest differs; check exact code, CSVs and NumPy version')
  for k,v in seed['records'].items():
   if k not in tasks or seed['seeds'][k]!=tasks[k]['seed']:raise ValueError('pilot seed mismatch '+k)
   check_result(k,v)
   if k not in done:done[k]=v
  atom_json(cp,done)
 if not set(done)<=set(tasks):raise ValueError('unknown checkpoint keys')
 for k,v in done.items():check_result(k,v)
 print(f'Archived reuse={len(reused)}, completed={len(done)}, remaining={len(tasks)-len(done)}',flush=True)
 if a.verify_reused:
  selected=[(k,r) for k,r in reused.items() if k.startswith('impact_epsilon_and_T') and int(r['T'])==1000 and int(r['run'])==0 and float(r['epsilon'])==0.1]
  assert len(selected)==4
  for k,r in selected:
   _,v=run((k,r))
   for name,val in v.items():
    if not math.isclose(val,float(r[name]),rel_tol=1e-8,abs_tol=1e-9):raise RuntimeError(f'reuse regression failure {k} {name}: {val} vs {r[name]}')
  print('Four unchanged algorithm/convexity pilot runs reproduce archived values.',flush=True)
 pending=[(k,r) for k,r in tasks.items() if k not in done];pending.sort(key=lambda kv:(int(kv[1]['T']),int(kv[1]['num_nodes']),int(kv[1]['run']),kv[0]))
 if a.max_runs is not None:pending=pending[:a.max_runs]
 if pending:
  with ProcessPoolExecutor(max_workers=a.workers) as pool:
   futures=[pool.submit(run,t) for t in pending]
   for future in as_completed(futures):
    k,v=future.result();done[k]=v;atom_json(cp,done);print(f'{len(done)}/{len(tasks)} {k}',flush=True)
 if len(done)!=len(tasks):
  print('Checkpoint only; no final CSV or figures published.',flush=True);return
 dest=out/'complete';dest.mkdir(exist_ok=True)
 for f in FILES:
  temp=dest/(f+'.tmp')
  with temp.open('w',newline='') as h:
   w=csv.DictWriter(h,fieldnames=fields[f]);w.writeheader()
   for i,r in enumerate(rows[f]):
    k=f+':'+str(i);s=dict(r)
    if k in done:
     v=done[k];s.update({m:v.get(m,'') for m in METRICS},implementation='query-feasible-cap09-v2' if len(v)==3 else 'query-feasible-cap09-v2-regret-only')
    w.writerow(s)
  temp.replace(dest/f)
 subprocess.run([sys.executable,'-m','adv_setting.plot_adv_setting_results','--output_dir',str(dest),'--recompute_summary','--require_runs','10'],cwd=ROOT,check=True)
 print('Complete candidate runs and figures:',dest)
if __name__=='__main__':main()
