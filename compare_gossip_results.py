"""Compare normalized regret using identical archived run keys and seeds."""
import argparse, csv, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from adv_setting.adv_setting_common import RUN_KEY_FIELDS, row_key, validate_run_rows

FILES=[f'{e}_alpha{a}_runs.csv' for e in ('impact_epsilon_and_T','impact_topology') for a in (0,1)]

def write(path,rows):
    if not rows: raise ValueError('empty comparison')
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('old','new','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if any(a.output.resolve()==d.resolve() or d.resolve() in a.output.resolve().parents for d in (a.old,a.new)):
        raise ValueError('comparison output must be separate from inputs')
    a.output.mkdir(parents=True,exist_ok=True)
    result=[]; by_setting=defaultdict(dict)
    for f in FILES:
        tables=[]
        for d in (a.old,a.new):
            with (d/f).open(newline='') as h:rows=list(csv.DictReader(h))
            errors=validate_run_rows(rows,expected_runs=10,base_seed=20261109)
            if errors:raise ValueError(f'{d/f}: {errors[:5]}')
            table={row_key(r,RUN_KEY_FIELDS+('run','seed')):r for r in rows}
            if len(table)!=len(rows):raise ValueError('duplicate run key')
            tables.append(table)
        old,new=tables
        if old.keys()!=new.keys():raise ValueError('run keys or seeds differ: '+f)
        groups=defaultdict(list)
        for k,r in old.items():
            groups[k[:-2]].append((float(r['avg_regret_per_round']),float(new[k]['avg_regret_per_round'])))
        for k,pairs in sorted(groups.items()):
            x,y=np.array(pairs,dtype=float).T
            if len(x)!=10 or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('invalid comparison group')
            ox,ny=float(x.mean()),float(y.mean())
            # Absolute signed differences remain meaningful when the denominator is zero.
            rec=dict(zip(RUN_KEY_FIELDS,k))
            rec.update(old_mean=ox,new_mean=ny,old_sd=float(x.std(ddof=1)),new_sd=float(y.std(ddof=1)),
                       paired_delta_mean=float((y-x).mean()),paired_delta_sd=float((y-x).std(ddof=1)),
                       relative_change_pct=100*(ny-ox)/ox if ox>0 else '')
            result.append(rec)
            by_setting[tuple(v for i,v in enumerate(k) if RUN_KEY_FIELDS[i]!='algorithm')][rec['algorithm']]=(ox,ny)
    gains=[]
    for k,algs in sorted(by_setting.items()):
        proposed=[v for n,v in algs.items() if n.startswith('PTF-') and 'fixed-block' not in n]
        vanilla=[v for n,v in algs.items() if n.startswith('PTF-') and 'fixed-block' in n]
        if len(proposed)!=1 or len(vanilla)!=1:continue
        x,y=proposed[0]; u,v=vanilla[0]
        rec=dict(zip([n for n in RUN_KEY_FIELDS if n!='algorithm'],k))
        rec.update(old_proposed=x,new_proposed=y,old_vanilla=u,new_vanilla=v,
                   old_improvement_pct=100*(1-x/u) if u>0 else '',
                   new_improvement_pct=100*(1-y/v) if v>0 else '',
                   old_win=int(x<u),new_win=int(y<v))
        gains.append(rec)
    if len(gains)!=84:raise ValueError(f'expected 84 proposed/Vanilla comparisons, got {len(gains)}')
    write(a.output/'paired_regret_comparison.csv',result)
    write(a.output/'vanilla_comparison.csv',gains)
    wins=[]
    for alpha in ('0','1'):
        rows=[r for r in gains if float(r['alpha_reg'])==float(alpha)]
        wins.append(dict(alpha_reg=alpha,settings=len(rows),old_wins=sum(r['old_win'] for r in rows),new_wins=sum(r['new_win'] for r in rows)))
    write(a.output/'vanilla_wins.csv',wins)
    print(wins)

if __name__=='__main__':main()
