"""Grouped bar version of the verified demographic comparison figure."""
from pathlib import Path
import hashlib
import json
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'manuscript/w3phiai27/figures/demographic_comparison'
OUT=BASE/'bar_charts'
OUT.mkdir(exist_ok=True)
verification=json.loads((BASE/'verification.json').read_text())
for path,digest in verification['source_sha256'].items():
    assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest
data=pd.read_csv(BASE/'figure2_source_values.csv')
assert len(data)==36 and not data.duplicated(['panel','configuration','arm']).any()
assert (data.lower<=data.auc).all() and (data.auc<=data.upper).all()
colors=['#0072B2','#D55E00','#008060']
arms=['broad','restricted','matched']
configs=['age_only','full','no_age','sensors']
labels=['Age\nonly','Full\ninputs','Full\n(no age)','Sensors\nonly']
titles=['A  Cohort comparison','B  Equal-size cohorts','C  Common evaluation']
subtitles=['N = 2,431 / 1,243 / 702','N = 702; same 351 PD cases','Same 702 matched participants']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
 'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
handles=[Patch(facecolor=c,label=l) for c,l in zip(colors,['Broad adult','Age 35-75','Age/sex matched'])]

def draw(ax,panel,standalone=False):
    for j,arm in enumerate(arms):
        d=data[(data.panel==panel)&(data.arm==arm)].set_index('configuration').loc[configs]
        x=np.arange(4)+(j-1)*.24
        ax.bar(x,d.auc,width=.22,color=colors[j],edgecolor='white',linewidth=.35,zorder=3)
        # Vertical labels keep all 12 exact estimates readable without collisions.
        for xx,r in zip(x,d.itertuples()):
            ax.text(xx,r.auc+.013,f'{r.auc:.3f}',ha='center',va='bottom',rotation=90,
                    fontsize=8 if standalone else 6.6)
    ax.set_ylim(0,1.13)
    ax.set_yticks(np.arange(0,1.01,.2))
    ax.set_xticks(np.arange(4),labels,fontsize=8 if standalone else 7)
    ax.tick_params(axis='x',length=0,pad=6)
    ax.tick_params(axis='y',labelsize=7)
    ax.grid(axis='y',color='#dde2e7',linewidth=.5,zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(.5,color='#555555',ls='--',lw=.7,zorder=2)
    ax.set_title(titles[ord(panel)-65],loc='left',fontweight='bold',fontsize=10 if standalone else 8.5,pad=23)
    ax.text(0,1.025,subtitles[ord(panel)-65],transform=ax.transAxes,fontsize=7)
    ax.set_ylabel('ROC-AUC',fontsize=8)
    ax.set_xlim(-.55,3.55)

fig,axes=plt.subplots(1,3,figsize=(7.2,3.45))
fig.subplots_adjust(left=.065,right=.99,bottom=.24,top=.76,wspace=.31)
for ax,panel in zip(axes,'ABC'):draw(ax,panel)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.99),ncol=3,frameon=False,fontsize=8)
fig.text(.065,.045,'A: AUC estimates.  B-C: means of 20 draws; matched reference fixed.\nDashed line: AUC = 0.5. All bars start at zero; all panels use the same scale.',fontsize=6.8,linespacing=1.5)
for ext in ['pdf','png','svg']:fig.savefig(OUT/f'figure2_demographic_bars.{ext}',dpi=450)
plt.close(fig)
for panel in 'ABC':
    fig,ax=plt.subplots(figsize=(4.8,3.8))
    fig.subplots_adjust(left=.13,right=.98,bottom=.24,top=.76)
    draw(ax,panel,True)
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,fontsize=8)
    fig.text(.13,.045,'AUC estimates; uncertainty not displayed.' if panel=='A' else 'Mean of 20 draws; matched reference fixed.',fontsize=8)
    for ext in ['pdf','png']:fig.savefig(OUT/f'panel_{panel}_bars.{ext}',dpi=450)
    plt.close(fig)
caption='Demographic comparisons using the fixed demographic selector. Panel A changes population and sample size; bars show ROC-AUC estimates. Panels B-C hold PD identities, class sizes and sex counts fixed. Panel B evaluates each sampled population; Panel C uses the same matched evaluation population. Broad and age-restricted bars show means over all 20 draws. Confidence intervals and draw ranges are not displayed. The matched reference is fixed. All panels share the same zero-based scale; the dashed line marks AUC 0.5.'
(OUT/'caption.txt').write_text(caption+'\n')
(OUT/'overleaf_figure2.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figure2_demographic_bars.pdf}\n\\caption{'+caption.replace('%','\\%')+'}\n\\label{fig:demographics}\n\\end{figure*}\n')
data.to_csv(OUT/'source_values.csv',index=False)
(OUT/'verification.json').write_text(json.dumps({'source_sha256':verification['source_sha256'],
 'verified_estimates':36,'unchanged_from_verified_dot_plot':True,
 'uncertainty_displayed':False,'all_bar_baselines':0,'shared_axis_limits':[0,1.13]},indent=2)+'\n')
with zipfile.ZipFile(BASE.parent/'Figure2_demographic_bar_graphs.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,p.name)
print(OUT)
