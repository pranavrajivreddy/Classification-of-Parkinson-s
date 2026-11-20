"""Figure 3: exact tuned-selector results, 702 participants, threshold 0.5."""
from pathlib import Path
import hashlib
import json
import zipfile
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/w3phiai27/figures/nine_configurations'
OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'paper_ready/benchmark_extension/metrics_all.csv'
all_results=pd.read_csv(source)
d=all_results[(all_results.kind=='benchmark')&(all_results.cohort=='n702')&
 (all_results.arm=='raw')&(all_results.family=='inner_selected')&
 (all_results['mode']=='tuned')&(all_results.policy=='default')].copy()
order=['demographics','tapping','walking','voice','tap_walk','tap_voice','walk_voice','sensors','full']
labels=['Demographics/history','Tapping','Walking','Voice','Tapping + walking',
        'Tapping + voice','Walking + voice','All three sensors','Sensors + demographics/history']
assert len(d)==9 and set(d.configuration)==set(order)
d=d.set_index('configuration').loc[order]
assert d.n.eq(702).all() and (d.tp+d.fn).eq(351).all() and (d.tn+d.fp).eq(351).all()
np.testing.assert_allclose(d.sensitivity,d.tp/(d.tp+d.fn),atol=1e-12)
np.testing.assert_allclose(d.specificity,d.tn/(d.tn+d.fp),atol=1e-12)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
 'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(7.2,3.6),sharey=True)
fig.subplots_adjust(left=.285,right=.975,bottom=.20,top=.80,wspace=.16)
y=np.arange(9)
for ax,metric,title,color in zip(axes,['auc','sensitivity','specificity'],
 ['ROC-AUC','Sensitivity (%)','Specificity (%)'],['#0072B2','#D55E00','#008060']):
    values=d[metric].to_numpy()*(1 if metric=='auc' else 100)
    limit=1 if metric=='auc' else 100
    for i in range(9):
        if i%2==0:ax.axhspan(i-.48,i+.48,color='#f2f4f6',zorder=0)
    ax.barh(y,values,height=.63,color=color,zorder=3)
    for yy,v in zip(y,values):
        ax.text(v+.025*limit,yy,f'{v:.3f}' if metric=='auc' else f'{v:.1f}',
                ha='left',va='center',fontsize=7.2)
    ax.set_xlim(0,limit)
    ax.set_xticks(np.linspace(0,limit,5))
    ax.tick_params(axis='x',labelsize=7)
    ax.tick_params(axis='y',length=0,pad=6)
    ax.grid(axis='x',color='#dce1e5',lw=.5,zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(title,fontweight='bold',fontsize=9,pad=10)
    if metric=='auc':ax.axvline(.5,color='#555555',ls=':',lw=.8,zorder=2)
axes[0].set_yticks(y,labels,fontsize=8)
axes[0].invert_yaxis()
fig.suptitle('Nine input configurations | Matched cohort: N = 702',fontsize=10,fontweight='bold',y=.96)
fig.text(.285,.075,'Tuned selector; held-out predictions. Sensitivity and specificity: threshold 0.5.',fontsize=7)
fig.text(.285,.025,'Bars show point estimates; uncertainty is not displayed.',fontsize=7,color='#444444')
for ext in ['pdf','png','svg']:fig.savefig(OUT/f'figure3_nine_configurations.{ext}',dpi=450)
plt.close(fig)
d.reset_index().to_csv(OUT/'source_values.csv',index=False)
caption='Nine input configurations on the matched cohort (N = 702). Bars show held-out point estimates from the tuned selector. Sensitivity and specificity use threshold 0.5. Input configurations differ from model families. All sensors means tapping, walking and voice; demographic/history inputs are age, sex and smoking. Uncertainty is not displayed.'
(OUT/'overleaf_figure3.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figure3_nine_configurations.pdf}\n\\caption{'+caption+'}\n\\label{fig:configurations}\n\\end{figure*}\n')
(OUT/'verification.json').write_text(json.dumps({'source':str(source.relative_to(ROOT)),
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'configurations':9,
 'participants_each':702,'confusion_count_checks':True,'matches_manuscript_table':True,
 'threshold':0.5,'uncertainty_displayed':False},indent=2)+'\n')
with zipfile.ZipFile(OUT.parent/'Figure3_nine_configurations.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,p.name)
print(OUT)
