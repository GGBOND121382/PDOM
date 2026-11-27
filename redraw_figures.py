"""Reproduce the old 4-curve style, fixed canvas/axes, using unmodified summary CSVs."""
import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size':18,'axes.labelsize':18,'xtick.labelsize':18,'ytick.labelsize':18,'legend.fontsize':18})

def draw(src,dest,alpha,experiment,share):
    name=f'{experiment}_alpha{alpha}'
    with (src/f'{name}_summary.csv').open() as f: rows=list(csv.DictReader(f))
    method='FTAL' if alpha else 'FTRL'
    raw=['BDOO','PBDOO-strong' if alpha else 'PBDOO-cor1','PTF-PBDOO-fixed-block-strong' if alpha else 'PTF-PBDOO-fixed-block-optimized-eta-cor1','PTF-PBDOO-strong' if alpha else 'PTF-PBDOO-cor1']
    label=['BDOO','PBDOO',rf'PB$^2$D-{method} (Vanilla-block)',rf'PB$^2$D-{method}']
    styles=[('red','o','-'),('gold','s','-'),('black' if alpha else 'tab:purple','^' if alpha else 'D','--' if alpha else '-.'),('blue','D' if alpha else '^','-')]
    topology=experiment=='impact_topology'
    panels=['cycle','grid','complete'] if topology else [0.1,1.,10.]
    captions=['Cycle','Grid','All-connected'] if topology else [rf'$\epsilon$={v:g}' for v in panels]
    n=36 if topology else 9
    fig,axes=plt.subplots(1,3,figsize=(17.725,5.67),dpi=200,sharey=share)
    # Exact canvas across all figures; deliberately avoid bbox_inches='tight'.
    fig.subplots_adjust(left=.047,right=.994,bottom=.185,top=.891,wspace=.25)
    for j,(ax,p,caption) in enumerate(zip(axes,panels,captions)):
        for alg,(color,marker,ls) in zip(raw,styles):
            rs=[r for r in rows if r['algorithm']==alg and (r['topology']==p if topology else (alg=='BDOO' or (r['epsilon']!='' and float(r['epsilon'])==p)))]
            rs.sort(key=lambda r:int(r['T']))
            assert len(rs)==7 and all(int(r['num_runs'])==10 and int(r['num_nodes'])==n for r in rs),(name,p,alg)
            ax.errorbar([int(r['T']) for r in rs],[float(r['avg_regret_mean'])/n for r in rs],yerr=[float(r['avg_regret_std'])/n for r in rs],color=color,marker=marker,linestyle=ls,capsize=3)
        ax.set_xscale('log'); ax.set_xlabel('Learning time T');
        if not share or j==0: ax.set_ylabel('Normalized regret loss')
        if share and j: ax.tick_params(axis='y',labelleft=False)
        ax.tick_params(axis='both',labelsize=18)
        fig.text((ax.get_position().x0+ax.get_position().x1)/2,.036,f'({chr(97+j)}) {caption}',ha='center',va='center',fontsize=18)
    handles=[Line2D([0],[0],color=c,marker=m,linestyle=ls,label=l) for l,(c,m,ls) in zip(label,styles)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.99),ncol=4,frameon=True,borderaxespad=0)
    dest.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest/f'{name}_3subplots.png',dpi=200)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('summary_dir',type=Path);p.add_argument('output_dir',type=Path);a=p.parse_args()
    for alpha in (0,1):
        for experiment in ('impact_epsilon_and_T','impact_topology'):
            draw(a.summary_dir,a.output_dir/'old_style',alpha,experiment,experiment=='impact_topology')
            draw(a.summary_dir,a.output_dir/'shared_y',alpha,experiment,True)
