"""Figure 4 from saved held-out attribution and supported scenario summaries."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/w3phiai27/figures/interpretation'
OUT.mkdir(parents=True,exist_ok=True)
files=[ROOT/'paper_ready/explainability/heldout_shap_importance.csv',
 ROOT/'paper_ready/causal_extension/structural_scenario_results.csv']
shap=pd.read_csv(files[0]).sort_values('mean_absolute_shap',ascending=False).head(6)
all_s=pd.read_csv(files[1])
scenarios=['tapping_time','walking_cadence','walking_step_length']
s=all_s[(all_s.method=='expanded_selector')&(all_s.level==.1)&
 (all_s.support_set=='common_support')&(all_s.group=='PD')].set_index('scenario').loc[scenarios]
assert len(s)==3 and s.n.tolist()==[30,220,248]
np.testing.assert_allclose(s.n+s.excluded,351)
names={'tap_alternation_rate':'Tapping alternation rate','tap_flight_max':'Maximum inter-tap interval',
 'tap_flight_mean':'Mean inter-tap interval','tap_count_valid':'Valid tap count',
 'tap_flight_std':'Inter-tap interval\nstandard deviation','voice_f0_mean':'Mean voice\nfundamental frequency'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,
 'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def draw_a(ax):
    y=np.arange(6)
    ax.barh(y,shap.mean_absolute_shap,height=.60,color='#0072B2',zorder=3)
    for yy,v in zip(y,shap.mean_absolute_shap):ax.text(v+.0013,yy,f'{v:.5f}',va='center',fontsize=7)
    ax.set_yticks(y,[names[x] for x in shap.feature],fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0,.082)
    ax.set_xticks([0,.02,.04,.06,.08])
    ax.tick_params(axis='both',labelsize=7)
    ax.tick_params(axis='y',length=0)
    ax.grid(axis='x',color='#dce1e5',lw=.5)
    ax.set_axisbelow(True)
    ax.set_xlabel('Mean absolute SHAP\n(classifier-score units)',fontsize=7)
    ax.set_title('A  Top six feature contributions',fontsize=9,fontweight='bold',loc='left',pad=29)
    ax.text(0,1.045,'Demographic selector; N = 702',transform=ax.transAxes,fontsize=7)
def signed(v):return f'{v:+.3f}'.replace('-', '\N{MINUS SIGN}')
def draw_b(ax):
    for i,(label,r) in enumerate(zip(['Tapping rate','Walking cadence','Walking step length'],s.itertuples())):
        v=r.mean_score_change*100
        ax.barh(i,v,height=.28,color='#008060' if v<0 else '#D55E00',zorder=3)
        ax.text(-3.45,i-.29,f'{label} +10%  |  PD n = {r.n}',fontsize=7.5,va='center')
        ax.text(-3.45,i+.30,f'{signed(v)} pp; 95% CI [{signed(100*r.ci_low)}, {signed(100*r.ci_high)}]',fontsize=6.8,va='center')
    ax.set_ylim(2.55,-.6)
    ax.set_xlim(-3.5,.7)
    ax.set_xticks([-3,-2,-1,0])
    ax.set_yticks([])
    ax.tick_params(axis='x',labelsize=7)
    ax.axvline(0,color='#555555',lw=.9,zorder=2)
    ax.grid(axis='x',color='#dce1e5',lw=.5)
    ax.set_axisbelow(True)
    ax.set_xlabel('Change in classifier score\n(percentage points, pp)',fontsize=7)
    ax.set_title('B  Supported measurement scenarios',fontsize=9,fontweight='bold',loc='left',pad=29)
    ax.text(0,1.045,'Tuned selector; conditional 95% intervals in text',transform=ax.transAxes,fontsize=7)

fig=plt.figure(figsize=(7.2,3.45))
a=fig.add_axes([.205,.24,.285,.53]); b=fig.add_axes([.615,.24,.37,.53])
draw_a(a);draw_b(b)
fig.text(.205,.04,'Panels explain different fitted procedures. Score changes are not treatment effects.\nScenario rows use different supported PD subsets; intervals condition on fitted models.',fontsize=7,linespacing=1.4)
for ext in ['pdf','png','svg']:fig.savefig(OUT/f'figure4_interpretation.{ext}',dpi=450)
plt.close(fig)
for panel,func in [('A',draw_a),('B',draw_b)]:
    fig,ax=plt.subplots(figsize=(5.4,3.8))
    fig.subplots_adjust(left=.35 if panel=='A' else .12,right=.98,bottom=.20,top=.76)
    func(ax)
    for ext in ['pdf','png']:fig.savefig(OUT/f'panel_{panel}.{ext}',dpi=450)
    plt.close(fig)
shap.to_csv(OUT/'shap_source_values.csv',index=False)
s.reset_index().to_csv(OUT/'scenario_source_values.csv',index=False)
caption='Feature contributions and constrained prediction changes answer different questions. (A) Top six mean absolute held-out SHAP contributions for the demographic selector on 702 participants. (B) Mean classifier-score changes for the tuned selector under coordinated +10\\% measurement scenarios; conditional 95\\% intervals are printed beside the bars. Each scenario includes only PD participants supported at all four nonzero magnitudes in that scenario family. Different rows can represent different people. Score changes are percentage points, not changes in disease risk or treatment effects. The panels explain different fitted procedures.'
(OUT/'overleaf_figure4.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figure4_interpretation.pdf}\n\\caption{'+caption+'}\n\\label{fig:interpretation}\n\\end{figure*}\n')
(OUT/'verification.json').write_text(json.dumps({'source_sha256':{str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
 'shap_top_features':6,'shap_model':'demographic selector','scenario_model':'expanded tuned selector',
 'scenario_level':.1,'support_set':'common_support','group':'PD','n':s.n.tolist(),
 'matches_manuscript_scenario_table':True,'score_units':'percentage points (100 times probability-score difference)',
 'error_marks_drawn':False,'conditional_intervals_printed':True},indent=2)+'\n')
with zipfile.ZipFile(OUT.parent/'Figure4_interpretation_graphs.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,p.name)
print(OUT)
