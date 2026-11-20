"""Render Figure 2 from aggregate results only; no model fitting or data changes."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'manuscript/w3phiai27/figures/demographic_comparison'
OUT.mkdir(parents=True, exist_ok=True)
sources = [ROOT/'paper_ready/cohort_comparison/auc_intervals.csv',
           ROOT/'paper_ready/controlled_followup/cohort_auc_summary.csv',
           ROOT/'paper_ready/controlled_followup/cohort_metrics_all.csv']
a, s, raw = [pd.read_csv(p) for p in sources]
raw = raw[raw.policy.eq('default')]
# Verify every mean and range against all saved draws, rather than rounded prose.
for r in s.itertuples():
    v = raw[(raw.arm == r.arm) & (raw.configuration == r.configuration)
            & (raw.evaluation == r.evaluation)]
    assert len(v) == r.draws and v.seed.nunique() == r.draws
    assert v.n.eq(702).all()
    np.testing.assert_allclose([v.auc.mean(), v.auc.min(), v.auc.max()],
                               [r.auc_mean, r.auc_min, r.auc_max], atol=1e-12)

configs = ['age_only', 'full', 'no_age', 'sensors']
labels = ['Age only', 'Full inputs', 'Full without age', 'Sensors only']
arms = ['broad', 'restricted', 'matched']
colors = ['#0072B2', '#D55E00', '#008060']
markers = ['o', 's', 'D']
rows = []
for panel, ev in [('A', 'own_cohort'), ('B', 'own'), ('C', 'common')]:
    for c in configs:
        for arm in arms:
            if panel == 'A':
                r = a[(a.training_cohort == arm) & (a.configuration == c) & (a.evaluation == ev)].iloc[0]
                val, lo, hi, count = r.auc, r.ci_low, r.ci_high, 1
                uncertainty = 'conditional 95% bootstrap interval'
            else:
                # The fixed matched reference is the same evaluation in both panels.
                r = s[(s.arm == arm) & (s.configuration == c) &
                      (s.evaluation == ('common' if arm == 'matched' else ev))].iloc[0]
                val, lo, hi, count = r.auc_mean, r.auc_min, r.auc_max, int(r.draws)
                uncertainty = 'fixed reference; no draw range' if arm == 'matched' else 'minimum-maximum of 20 draws'
            rows.append(dict(panel=panel, configuration=c, arm=arm, auc=float(val),
                             lower=float(lo), upper=float(hi), draws=count, uncertainty=uncertainty))
data = pd.DataFrame(rows)
for c in configs:
    v = data[(data.arm == 'matched') & (data.configuration == c)].auc
    np.testing.assert_allclose(v, v.iloc[0], atol=1e-12)
data.to_csv(OUT/'figure2_source_values.csv', index=False)

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8,
 'axes.titlesize':9, 'axes.labelsize':8, 'xtick.labelsize':7,
 'ytick.labelsize':8, 'pdf.fonttype':42, 'ps.fonttype':42,
 'axes.spines.top':False, 'axes.spines.right':False})
titles = {'A':'A  Cohort comparison', 'B':'B  Equal-size cohorts', 'C':'C  Common evaluation'}
subtitles = {'A':'N = 2,431 / 1,243 / 702',
             'B':'N = 702; same 351 PD cases',
             'C':'Same 702 matched participants'}
handles = [Line2D([], [], color=co, marker=ma, lw=1.2, markersize=4, label=lab)
           for co,ma,lab in zip(colors,markers,['Broad adult', 'Age 35-75', 'Age/sex matched'])]

def draw(ax, panel, show_labels=True):
    for i,c in enumerate(configs):
        y = 3-i
        if i % 2 == 0:
            ax.axhspan(y-.43,y+.43, color='#f3f5f7', zorder=0)
        for j,arm in enumerate(arms):
            r = data[(data.panel == panel)&(data.configuration == c)&(data.arm == arm)].iloc[0]
            yy = y + (.24,0,-.24)[j]
            ax.errorbar(r.auc, yy, xerr=[[r.auc-r.lower],[r.upper-r.auc]],
                fmt=markers[j], color=colors[j], markersize=3.5,
                elinewidth=1.05, capsize=2, capthick=.8, zorder=3)
            ax.text(r.upper+.010, yy, f'{r.auc:.3f}', va='center', fontsize=6.5, color=colors[j])
    ax.set_yticks([3,2,1,0], labels if show_labels else ['']*4)
    ax.tick_params(axis='y', length=0, pad=5)
    ax.set_ylim(-.52,3.57)
    ax.set_xlim((.48,1.04) if panel in 'AB' else (.48,.85))
    ax.set_xticks([.5,.6,.7,.8,.9,1.] if panel in 'AB' else [.5,.6,.7,.8])
    ax.axvline(.5, ls=':', lw=.8, color='#777777', zorder=1)
    ax.grid(axis='x', color='#dce1e5', lw=.5)
    ax.set_axisbelow(True)
    ax.set_xlabel('ROC-AUC' + (' (expanded scale)' if panel == 'C' else ''))
    ax.set_title(titles[panel], loc='left', fontweight='bold', pad=23)
    ax.text(0,1.025,subtitles[panel], transform=ax.transAxes,fontsize=6.5)

fig, axes = plt.subplots(1,3,figsize=(7.0,3.05))
fig.subplots_adjust(left=.15,right=.995,bottom=.22,top=.77,wspace=.16)
for ax,panel in zip(axes,'ABC'): draw(ax,panel, panel=='A')
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.56,1.01),ncol=3,frameon=False,fontsize=8)
fig.text(.15,.035,'A: conditional 95% intervals.  B-C: mean and full range of 20 draws; matched reference fixed.',fontsize=6.7)
for ext in ['pdf','png','svg']: fig.savefig(OUT/f'figure2_demographic_comparison.{ext}',dpi=450)
plt.close(fig)
for panel in 'ABC':
    fig,ax=plt.subplots(figsize=(4.8,3.5))
    fig.subplots_adjust(left=.28,right=.98,bottom=.23,top=.77)
    draw(ax,panel)
    fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False,fontsize=7)
    fig.text(.28,.055,'Conditional 95% intervals' if panel=='A' else 'Mean and min-max over 20 draws; matched reference fixed.',fontsize=7)
    for ext in ['pdf','png']: fig.savefig(OUT/f'panel_{panel}.{ext}',dpi=450)
    plt.close(fig)

(OUT/'verification.json').write_text(json.dumps({'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
 'points':len(data), 'all_draw_summaries_recomputed':True,'matched_reference_consistent':True,
 'panel_C_expanded_axis':True,'participant_level_data_used':False},indent=2)+'\n')
(OUT/'figure2_caption.txt').write_text('Demographic comparisons using the fixed demographic selector. Panel A changes population and sample size; points show ROC-AUC and conditional 95% bootstrap intervals. Panels B-C hold PD identities, class sizes and sex counts fixed. Panel B evaluates each sampled population; Panel C evaluates all training strategies on the same matched population. For broad and age-restricted controls, points show the mean and whiskers the minimum-maximum across all 20 draws. These draw ranges are not confidence intervals. Matched results are a single fixed reference. Panel C uses an expanded horizontal scale. Full inputs include sensors and demographic/history variables.\n')
print('Verified all saved draw summaries and generated 36 plotted estimates:', OUT)
