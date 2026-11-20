"""Figure 5: feature-development comparisons, not configuration-search results."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'manuscript/w3phiai27/figures/agent_comparison'
OUT.mkdir(parents=True,exist_ok=True)
src=ROOT/'paper_ready/benchmark_extension/paired_comparisons.json'
j=json.loads(src.read_text())
rows=[]
for c in j['comparisons']:
    if c['comparison']=='agent_minus_raw_full':
        r=c['conditional_bootstrap']['auc']
        rows.append({'family':c['family'],**r})
order=['lr','rf','xgboost','lightgbm','catboost','mlp','inner_selected']
a=pd.DataFrame(rows).set_index('family').loc[order]
assert len(a)==7 and ((a.low<0)&(a.high>0)).all()
np.testing.assert_allclose(a.estimate_a-a.estimate_b,a.difference,atol=1e-12)
sources=[src]; bs=[]
for folder,name,agent,n in [('final_multiagent','DeepSeek','multiagent',492),
 ('claude_single_agent','Claude','claude',492),('claude_reverse_split','Claude: larger discovery','claude',210)]:
    p=ROOT/f'paper_ready/{folder}/metrics_all.csv';sources.append(p)
    d=pd.read_csv(p);d=d[(d.method=='equal_ensemble')&(d.policy=='default')].set_index('arm').loc[['raw',agent,'automatic']]
    assert len(d)==3 and d.n.eq(n).all()
    np.testing.assert_allclose(d.auc,d.auc.iloc[0],atol=1e-12)
    for arm,r in d.iterrows():bs.append({'comparison':name,'procedure':{'raw':'Conventional','automatic':'Automatic',agent:'Agent'}[arm],
        'n':n,'auc':r.auc,'method':'equal_ensemble'})
b=pd.DataFrame(bs)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'ps.fonttype':42,
 'axes.spines.top':False,'axes.spines.right':False})
names=['Logistic regression','Random forest','XGBoost','LightGBM','CatBoost','Multilayer perceptron','Tuned selector']
def sign(v):return f'{v:+.4f}'.replace('-', '\N{MINUS SIGN}')
def panel_a(ax):
    for i,r in enumerate(a.itertuples()):
        if i%2==0:ax.axhspan(i-.32,i+.52,color='#f2f4f6',zorder=0)
        ax.barh(i,r.difference,height=.30,color='#0072B2',zorder=3)
        ax.text(r.difference+(.001 if r.difference>=0 else -.001),i,sign(r.difference),
                ha='left' if r.difference>=0 else 'right',va='center',fontsize=7)
        ax.text(-.028,i+.34,f'95% CI [{sign(r.low)}, {sign(r.high)}]',va='center',fontsize=6.3,color='#444444')
    ax.set_yticks(np.arange(7),names,fontsize=7)
    ax.tick_params(axis='y',length=0)
    ax.set_ylim(6.65,-.6);ax.set_xlim(-.029,.029)
    ax.set_xticks([-.02,0,.02]);ax.tick_params(axis='x',labelsize=7)
    ax.axvline(0,color='#555555',lw=.8,zorder=2)
    ax.set_xlabel('AUC difference: agent - conventional',fontsize=7)
    ax.set_title('A  Feature effects across learners',loc='left',fontweight='bold',fontsize=9,pad=29)
    ax.text(0,1.04,'Full inputs; evaluation N = 492',transform=ax.transAxes,fontsize=7)
def panel_b(ax):
    colors=['#0072B2','#D55E00','#008060'];procs=['Conventional','Agent','Automatic']
    groups=b.comparison.drop_duplicates().tolist()
    for k,(p,c) in enumerate(zip(procs,colors)):
        vals=[b[(b.comparison==g)&(b.procedure==p)].auc.iloc[0] for g in groups]
        x=np.arange(3)+(k-1)*.24
        ax.bar(x,vals,width=.22,color=c,zorder=3)
        for xx,v in zip(x,vals):ax.text(xx,v+.015,f'{v:.4f}',ha='center',va='bottom',rotation=90,fontsize=7)
    ax.set_xticks(range(3),['DeepSeek\nN = 492','Claude\nN = 492','Claude: larger\ndiscovery\nN = 210'],fontsize=7)
    ax.set_ylim(0,1.13);ax.set_yticks([0,.2,.4,.6,.8,1]);ax.tick_params(axis='y',labelsize=7)
    ax.grid(axis='y',color='#dce1e5',lw=.5);ax.set_axisbelow(True)
    ax.set_ylabel('Pooled ROC-AUC',fontsize=8)
    ax.set_title('B  Scripted feature evaluations',loc='left',fontweight='bold',fontsize=9,pad=29)
    ax.text(0,1.04,'Equal-weight ensemble; predictions coincide',transform=ax.transAxes,fontsize=6.2)
    ax.legend(handles=[Patch(facecolor=c,label=p) for c,p in zip(colors,procs)],
              loc='upper center',ncol=3,fontsize=6,frameon=False,bbox_to_anchor=(.5,.99),handlelength=1,columnspacing=.7,handletextpad=.4)

fig=plt.figure(figsize=(7.2,4.3))
ax=fig.add_axes([.19,.25,.32,.56]);bx=fig.add_axes([.65,.25,.33,.56])
panel_a(ax);panel_b(bx)
fig.text(.19,.075,'A: conditional 95% intervals are printed, not drawn; all include zero.\nB: conventional inputs retained in all three comparisons. Different evaluation sets\nmean that absolute AUCs cannot rank agent quality.',fontsize=7,linespacing=1.5)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'figure5_agent_comparison.{ext}',dpi=450)
plt.close(fig)
for label,func in [('A',panel_a),('B',panel_b)]:
    fig,ax=plt.subplots(figsize=(5.3,4.3))
    fig.subplots_adjust(left=.30 if label=='A' else .14,right=.97,bottom=.20,top=.78)
    func(ax)
    for ext in ['png','pdf']:fig.savefig(OUT/f'panel_{label}.{ext}',dpi=450)
    plt.close(fig)
a.reset_index().to_csv(OUT/'feature_effects.csv',index=False)
b.to_csv(OUT/'scripted_evaluations.csv',index=False)
caption='Completed feature-development evaluations. Panel A shows all seven agent-minus-conventional AUC differences on 492 evaluation participants, with conditional 95\\% intervals printed beside the bars. Intervals are unadjusted across procedures and all include zero. Panel B shows pooled AUCs for the equal-weight ensembles in the DeepSeek, standard Claude and larger-discovery Claude comparisons. All retained conventional inputs; conventional, agent and automatic procedures therefore produced identical predictions within each comparison. Evaluation sets differ between designs, so absolute AUCs do not rank agent quality.'
(OUT/'overleaf_figure5.tex').write_text('\\begin{figure*}[t]\n\\centering\n\\includegraphics[width=\\textwidth]{figure5_agent_comparison.pdf}\n\\caption{'+caption+'}\n\\label{fig:agents}\n\\end{figure*}\n')
(OUT/'verification.json').write_text(json.dumps({'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
 'all_seven_effects_match_manuscript':True,'all_intervals_include_zero':True,
 'scripted_rows':len(b),'within_comparison_aucs_identical':True,'error_marks_drawn':False},indent=2)+'\n')
with zipfile.ZipFile(OUT.parent/'Figure5_agent_comparison.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.iterdir()):
        if p.is_file():z.write(p,p.name)
print(OUT)
