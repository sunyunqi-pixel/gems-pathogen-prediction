import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.ticker as mtick

OUT='figures'; os.makedirs(OUT,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.8,
                     'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,
                     'figure.facecolor':'white','axes.facecolor':'white'})
NAME={'shigella':'Shigella','st_etec':'ST-ETEC','rotavirus':'Rotavirus',
      'cryptosporidium':'Cryptosporidium','norovirus':'Norovirus'}
RAW=['random','site','country']
RLAB={'random':'Random','site':'Health-centre-held-out','country':'Country-held-out'}
RSHORT={'random':'Random','site':'Health centre','country':'Country'}
RC={'random':'#4C72B0','site':'#DD8452','country':'#C44E52'}
PATH_ORDER=['Shigella','ST-ETEC','Rotavirus','Cryptosporidium','Norovirus']
KEY={'Shigella':'shigella','ST-ETEC':'st_etec','Rotavirus':'rotavirus',
     'Cryptosporidium':'cryptosporidium','Norovirus':'norovirus'}
COL={'Shigella':'#4C72B0','ST-ETEC':'#C44E52','Rotavirus':'#55A868',
     'Cryptosporidium':'#8172B2','Norovirus':'#DD8452'}
FAMCOL={'temperature':'#C44E52','precipitation':'#4C72B0','humidity':'#55A868',
        'seasonality':'#8172B2','geography':'#CCB974','demographic_child':'#64B5CD',
        'household_wash_ses':'#DD8452'}
FAMLAB={'temperature':'Temperature','precipitation':'Precipitation','humidity':'Humidity',
        'seasonality':'Seasonality','geography':'Geography','demographic_child':'Demographic (child)',
        'household_wash_ses':'Household WASH / SES'}

t2=pd.read_csv('../table2_primary_performance_clean.csv')
t2['p']=t2.pathogen.map(NAME); assert t2.p.notna().all()
t2=t2.set_index(['p','regime'])

# ===== FIG 2 forest =====
ypos=[];ylab=[];meta=[];y=0;sep=[]
for p in PATH_ORDER:
    for rk in RAW:
        ypos.append(y); ylab.append(RSHORT[rk].replace('Health centre','Health\ncentre')); meta.append((p,rk)); y+=1
    sep.append(y-0.5); y+=0.7
fig,axes=plt.subplots(1,3,figsize=(11.2,6.2),sharey=True,
                      gridspec_kw={'width_ratios':[1.25,1.0,1.0]})
axA,axB,axC=axes
for (p,rk),yy in zip(meta,ypos):
    row=t2.loc[(p,rk)]; c=RC[rk]
    axA.plot([row.auc_ci_low,row.auc_ci_high],[yy,yy],color=c,lw=1.5,solid_capstyle='round',zorder=2)
    axA.plot(row.auc_weighted,yy,'o',color=c,ms=6,mec='white',mew=0.8,zorder=3)
    axB.plot([row.ap_ci_low,row.ap_ci_high],[yy,yy],color=c,lw=1.5,solid_capstyle='round',zorder=2)
    axB.plot(row.ap_weighted,yy,'o',color=c,ms=6,mec='white',mew=0.8,zorder=3)
    dv=row.delta_auc_vs_random_grouped_minus_random
    if pd.notna(dv):
        axC.plot([row.delta_auc_ci_low,row.delta_auc_ci_high],[yy,yy],color=c,lw=1.5,solid_capstyle='round',zorder=2)
        axC.plot(dv,yy,'o',color=c,ms=6,mec='white',mew=0.8,zorder=3)
        axC.text(dv,yy-0.40,'%.4f'%dv,ha='center',va='top',fontsize=6.4,color=c)
    else:
        axC.text(0.5,yy,'Reference',transform=axC.get_yaxis_transform(),ha='center',
                 va='center',fontsize=6.4,color='#888888',style='italic')
for s in sep:
    for ax in axes: ax.axhline(s,color='#DDDDDD',lw=0.7,zorder=1)
for i,p in enumerate(PATH_ORDER):
    yc=np.mean([ypos[i*3],ypos[i*3+2]])
    axA.text(-0.235,yc,p,transform=axA.get_yaxis_transform(),ha='right',va='center',
             fontsize=9.5,fontweight='bold')
for ax,xl in zip(axes,['AUC (95% CI)','Average precision (95% CI)',
                       '$\\Delta$AUC, grouped minus random (95% CI)']):
    ax.set_xlabel(xl,fontsize=8.6)
axA.set_yticks(ypos); axA.set_yticklabels(ylab,fontsize=7.6)
axA.set_xlim(0.58,0.90); axB.set_xlim(0.18,0.88); axC.set_xlim(-0.18,0.03)
axC.axvline(0,color='#555555',lw=1.0,ls='--',zorder=1)
axA.set_ylim(-0.9,max(ypos)+0.5)
axC.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))
axA.legend(handles=[Patch(facecolor=RC[k],label=RLAB[k]) for k in RAW],
           loc='lower left',fontsize=7.4,frameon=False)
fig.suptitle('Discrimination of the cleaned-feature weighted ensemble across validation regimes',
             fontsize=10.2,fontweight='bold',y=0.985)
fig.text(0.5,0.005,'Confidence intervals are percentile 95% intervals from 2,000 sentinel health centre cluster bootstrap resamples. '
         '$\\Delta$AUC has no counterpart for random cross-validation, which serves as the reference.',ha='center',
         fontsize=7,color='#555555')
plt.tight_layout(rect=[0.005,0.03,1,0.962])
plt.savefig(OUT+'/figure2_forest_auc_ap_delta.png',bbox_inches='tight'); plt.close()
print('[ok] figure2_forest_auc_ap_delta.png')

# ===== FIG 2b enhanced slope =====
fig,ax=plt.subplots(figsize=(6.8,4.7))
xs=[0,1,2]
for p in PATH_ORDER:
    a=[t2.loc[(p,rk),'auc_weighted'] for rk in RAW]
    lo=[t2.loc[(p,rk),'auc_ci_low'] for rk in RAW]
    hi=[t2.loc[(p,rk),'auc_ci_high'] for rk in RAW]
    a_=np.array(a); off=(PATH_ORDER.index(p)-2)*0.030
    ax.errorbar(np.array(xs)+off,a_,yerr=[a_-np.array(lo),np.array(hi)-a_],fmt='none',
                ecolor=COL[p],alpha=0.80,elinewidth=1.9,capsize=2.5,zorder=2)
    ax.plot(xs,a,'-o',color=COL[p],lw=1.9,ms=5.5,mec='white',mew=0.8,label=p,zorder=3)
    d=t2.loc[(p,'country'),'delta_auc_vs_random_grouped_minus_random']
    ax.annotate('$\\Delta$%.4f'%d,(2,a[2]),textcoords='offset points',xytext=(8,0),
                va='center',fontsize=7.4,color=COL[p],fontweight='bold')
ax.set_xticks(xs); ax.set_xticklabels([RSHORT[rk] for rk in RAW])
ax.set_xlim(-0.15,2.75); ax.set_ylabel('Pooled AUC'); ax.set_xlabel('Validation regime')
ax.legend(loc='lower left',fontsize=7.6,frameon=False,ncol=2)
ax.grid(axis='y',color='#EEEEEE',lw=0.7); ax.set_axisbelow(True)
ax.set_title('Transportability gradient by pathogen',fontsize=10.2,fontweight='bold')
fig.text(0.5,-0.02,'Vertical bars are percentile 95% confidence intervals from 2,000 sentinel health centre cluster bootstrap resamples. '
         'Labels show $\\Delta$AUC (country-held-out minus random).',ha='center',fontsize=7,color='#555555')
plt.tight_layout(); plt.savefig(OUT+'/figure2_slope_enhanced.png',bbox_inches='tight'); plt.close()
print('[ok] figure2_slope_enhanced.png')

# ===== FIG 4 case/control dot plot =====
t5=pd.read_csv('/home/user/uploaded_files/stage23/table5_recomputed_ci.csv')
t5['p']=t5.pathogen.map(NAME)
fig,ax=plt.subplots(figsize=(7.8,4.6))
ypos=[];ylab=[];yy=0
for p in PATH_ORDER:
    for sg,lab,mk in [('case','MSD cases','o'),('control','Controls','s')]:
        row=t5[(t5.p==p)&(t5.subgroup==sg)].iloc[0]
        cc='#C44E52' if sg=='case' else '#4C72B0'
        ax.plot([row.auc_ci_low,row.auc_ci_high],[yy,yy],color=cc,lw=2.0,alpha=0.55,solid_capstyle='round')
        ax.plot(row.auc,yy,mk,color=cc,ms=7,mec='white',mew=0.9,zorder=4)
        d=row.delta_auc_vs_mixed
        if abs(d)>=0.05:
            ax.annotate('%+.4f'%d,(row.auc,yy),textcoords='offset points',
                        xytext=(0,11 if d>0 else -19),ha='center',fontsize=7.2,
                        color=cc,fontweight='bold')
        ypos.append(yy); ylab.append(lab); yy+=1
    yy+=0.8
for i,p in enumerate(PATH_ORDER):
    y0=i*2+i*0.8; y1=y0+1
    ax.text(-0.15,(y0+y1)/2,p,transform=ax.get_yaxis_transform(),ha='right',va='center',
            fontsize=9,fontweight='bold')
    ax.axhline(y1+0.4,color='#DDDDDD',lw=0.7)
    m=t5[t5.p==p].mixed_auc.iloc[0]
    ax.plot([m,m],[y0-0.42,y1+0.42],ls='--',color='#666666',lw=1.1,zorder=2)
    ax.annotate('mixed %.4f'%m,(m,y0-0.42),textcoords='offset points',xytext=(-2,-15),
                ha='right',fontsize=6.4,color='#666666')
ax.set_yticks(ypos); ax.set_yticklabels(ylab,fontsize=7.8)
ax.set_xlim(0.54,0.90); ax.set_ylim(-1.55,yy-0.9); ax.set_xlabel('AUC (95% CI)')
ax.set_title('Case-only and control-only subgroup performance\n(cleaned-feature weighted ensemble, health-centre-held-out validation)',
             fontsize=9.6,fontweight='bold')
fig.text(0.5,-0.01,'Dashed vertical lines show the corresponding mixed-sample AUC. Values are annotated where the difference from the '
         'mixed-sample AUC is at least 0.05 in absolute value.',ha='center',fontsize=7,color='#555555')
plt.tight_layout(); plt.savefig(OUT+'/figure4_case_control_dot.png',bbox_inches='tight'); plt.close()
print('[ok] figure4_case_control_dot.png')

# ===== FIG 3 calibration 6-panel =====
cal=pd.read_csv('/home/user/uploaded_files/latest_outputs/calibration_summary_site_clean.csv').set_index('pathogen_label')
fig,axes=plt.subplots(2,3,figsize=(10.2,6.5))
axes=axes.ravel()
for i,p in enumerate(PATH_ORDER):
    ax=axes[i]
    d=pd.read_csv('../final_results_plots/%s_site_clean_binned_calibration.csv'%KEY[p])
    ax.plot([0,1],[0,1],ls=':',color='#999999',lw=1.0,zorder=1)
    ax.plot(d.pred_mean,d.obs_rate,'-o',color=COL[p],lw=1.7,ms=4.2,mec='white',mew=0.6,zorder=3)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    b=cal.loc[p]
    ax.set_title('%s\nAUC %.4f  |  slope %.2f'%(p,b.auc,b.calibration_slope_joint),fontsize=8.4)
    ax.tick_params(labelsize=7)
    if i>=3: ax.set_xlabel('Predicted probability',fontsize=7.8)
    if i%3==0: ax.set_ylabel('Observed proportion',fontsize=7.8)
ax=axes[5]
b=cal.loc[PATH_ORDER]
xs=np.arange(5); vals=b.brier_skill_score.values
lo=b.brier_skill_score_ci_low.values; hi=b.brier_skill_score_ci_high.values
ax.axhline(0,color='#555555',lw=1.0,ls='--',zorder=1)
ax.errorbar(xs,vals,yerr=[vals-lo,hi-vals],fmt='none',ecolor='#888888',elinewidth=1.2,capsize=3,zorder=2)
for x,v,c in zip(xs,vals,[COL[p] for p in PATH_ORDER]):
    ax.plot(x,v,'o',color=c,ms=7,mec='white',mew=0.9,zorder=3)
    ax.annotate('%.3f'%v,(x,v),textcoords='offset points',xytext=(0,10 if v>=0 else -16),
                ha='center',fontsize=6.8,color=c,fontweight='bold')
ax.set_xticks(xs); ax.set_xticklabels(PATH_ORDER,rotation=22,ha='right',fontsize=7.4)
ax.set_ylabel('Brier skill score',fontsize=7.8)
ax.set_title('Brier skill score\n(versus prevalence-only reference)',fontsize=8.4)
ax.tick_params(labelsize=7)
fig.suptitle('Pathogen-specific calibration of the cleaned-feature weighted ensemble (health-centre-held-out validation)',
             fontsize=10.2,fontweight='bold',y=0.99)
fig.text(0.5,0.005,'Panels show binned reliability curves on the pooled out-of-fold predictions; the diagonal denotes perfect calibration. '
         'The final panel shows Brier skill scores with 95% cluster bootstrap intervals; negative values indicate worse squared-error '
         'performance than a prevalence-only reference.',ha='center',fontsize=6.6,color='#555555')
plt.tight_layout(rect=[0,0.035,1,0.952])
plt.savefig(OUT+'/figure3_calibration_6panel.png',bbox_inches='tight'); plt.close()
print('[ok] figure3_calibration_6panel.png')

# ===== FIG 5 SHAP families =====
s=pd.read_csv('/home/user/uploaded_files/latest_outputs/shap_family_summary_long.csv')
piv=s.pivot(index='pathogen_label',columns='family',values='family_pct').reindex(PATH_ORDER).fillna(0.0)
order=[f for f in ['demographic_child','household_wash_ses','temperature','humidity',
                   'precipitation','geography','seasonality'] if f in piv.columns]
fig,ax=plt.subplots(figsize=(9.2,4.1))
left=np.zeros(len(piv)); ys=np.arange(len(piv))[::-1]
for f in order:
    v=piv[f].values
    ax.barh(ys,v,left=left,color=FAMCOL[f],edgecolor='white',linewidth=0.8,height=0.62,label=FAMLAB[f])
    for i,(x,w) in enumerate(zip(left,v)):
        if w>=5.5:
            ax.text(x+w/2,ys[i],'%.1f'%w,ha='center',va='center',fontsize=7.0,color='white',fontweight='bold')
    left=left+v
ax.set_yticks(ys); ax.set_yticklabels(piv.index,fontsize=9)
ax.set_xlim(0,100); ax.set_xlabel('Share of total absolute mean SHAP importance (%)')
ax.set_title('Feature-family SHAP contributions by pathogen',fontsize=10.2,fontweight='bold')
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.19),ncol=4,fontsize=7.4,frameon=False)
ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
fig.text(0.5,-0.17,'Contributions are from CatBoost surrogate models refitted to the full cleaned analytic datasets, expressed as percentages of the '
         'total absolute mean SHAP importance within each pathogen. The seven families were defined after initial inspection of the fitted models '
         'rather than a priori.',ha='center',fontsize=6.6,color='#555555')
plt.tight_layout(); plt.savefig(OUT+'/figure5_shap_feature_families.png',bbox_inches='tight'); plt.close()
print('[ok] figure5_shap_feature_families.png')
print()
for f in sorted(os.listdir(OUT)): print('  %-42s %8d bytes'%(f,os.path.getsize(OUT+'/'+f)))
