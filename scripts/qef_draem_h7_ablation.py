#!/usr/bin/env python3
"""
qef_draem_formal_corewide.py

Formal main-model run for selected Stage-1 winner: CoreWide + CrossAttn + LearnedGate.
Runs H1/H5/H7 with 5 seeds under rolling-origin evaluation.
"""
import random, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef, f1_score

DATA=Path(__file__).resolve().parent.parent / 'data' / 'dataset_with_qwen_event_memory.csv'
OUT=Path(__file__).resolve().parent.parent / 'results' / 'qwen_event_factor' / 'h7_ablation'; OUT.mkdir(parents=True,exist_ok=True)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
SEEDS=[42,123,777,2024,3407]; HORIZONS=[7]
HP=dict(d=32,dropout=.30,lr=8e-4,wd=1e-4,epochs=45,patience=7,batch=256,reg_w=.08)
NUM=['close','open','high','low','volume','amount','log_return','volatility_20d','volatility_60d','range_pct','return_lag1','return_lag3','return_lag5','volume_lag1','mkt_hbea_close','mkt_bea_close','mkt_shea_close','mkt_szea_close','mkt_cea_close','hs300_ret','eua_ret','blt_oil_ret','dq_oil_ret','coal_ret','cpi','m2']
MODULES=['policy','market','finance','trade','compliance']
AVAIL=['global_has','global_cnt','doc_mask','quality_score','importance_score','policy_has','market_has','finance_has','trade_has','compliance_has','qef_global_active_modules','qef_global_abs_signal','qef_global_signal_sum','qef_memory_signal_sum_ewm7','qef_memory_disagreement_ewm7']
CORE=['_signal','_pos_signal','_neg_signal','_abs_signal','_uncertainty','_signal_ewm7','_signal_ewm14','_pos_signal_ewm7','_neg_signal_ewm7','_abs_signal_ewm7','_active','_active_roll7']
WIDE=['_signal','_pos_signal','_neg_signal','_abs_signal','_uncertainty','_signal_ewm3','_signal_ewm7','_signal_ewm14','_signal_ewm21','_signal_lag1','_signal_lag3','_signal_lag5','_abs_signal_ewm7','_abs_signal_ewm14','_active','_active_roll3','_active_roll7','_active_roll14']
GLOBAL_CORE=['qef_global_signal_mean','qef_global_signal_sum','qef_global_abs_signal','qef_global_disagreement','qef_global_signal_sum_ewm7','qef_global_signal_sum_ewm14','qef_global_abs_signal_ewm7','qef_memory_signal_mean_ewm7','qef_memory_signal_sum_ewm7','qef_memory_disagreement_ewm7','qef_global_active_modules','qef_global_active_modules_roll7']
GLOBAL_WIDE=['qef_global_abs_signal','qef_global_disagreement','qef_global_signal_mean','qef_global_signal_sum','qef_global_abs_signal_ewm3','qef_global_abs_signal_ewm7','qef_global_abs_signal_ewm14','qef_global_abs_signal_ewm21','qef_global_signal_sum_ewm3','qef_global_signal_sum_ewm7','qef_global_signal_sum_ewm14','qef_global_signal_sum_ewm21','qef_global_signal_sum_lag1','qef_global_signal_sum_lag3','qef_global_active_modules','qef_global_active_modules_roll3','qef_global_active_modules_roll7','qef_global_active_modules_roll14']

def seed(s=42):
 random.seed(s); np.random.seed(s); torch.manual_seed(s)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
def ex(df,cols): return [c for c in cols if c in df.columns]
def groups(df,variant):
 if variant=='avail_only': return {'availability':ex(df,AVAIL)}
 out={}
 keys=WIDE if variant in ['core_wide','full_signal'] else CORE
 for m in MODULES:
  if variant=='full_signal': cols=sorted([c for c in df.columns if c.startswith(f'qef_{m}_') and ('signal' in c or 'active' in c or 'uncertainty' in c)])
  else: cols=[f'qef_{m}{k}' for k in keys if f'qef_{m}{k}' in df.columns]
  if cols: out[m]=cols
 out['global']=ex(df, GLOBAL_WIDE if variant in ['core_wide','full_signal'] else GLOBAL_CORE)
 return {k:v for k,v in out.items() if v}
def folds(df):
 yrs=sorted(df.date.dt.year.unique()); fs=[]
 for ty in yrs:
  vy=ty-1; tr=[y for y in yrs if y<vy]
  if ty<2021 or vy not in yrs or len(tr)<3: continue
  fs.append(dict(fold=f'Y{ty}',train=tr,val=vy,test=ty))
 return fs
def split(df,f):
 y=df.date.dt.year
 return df.index[y.isin(f['train'])].values,df.index[y==f['val']].values,df.index[y==f['test']].values
def build(df,idx,num,grp,avail,target):
 cols=num+[c for v in grp.values() for c in v]+avail+[target]
 sub=df.loc[idx,cols].replace([np.inf,-np.inf],np.nan).dropna(subset=[target]).copy(); y=sub[target].astype(float).values.astype('float32'); keep=y!=0
 sub=sub.iloc[np.where(keep)[0]]; y=y[keep]
 return sub[num].fillna(0).values.astype('float32'), {k:sub[v].fillna(0).values.astype('float32') for k,v in grp.items()}, sub[avail].fillna(0).values.astype('float32') if avail else np.zeros((len(sub),1),'float32'), y, sub.index.values
def scalefit(xs,xg,xa): return StandardScaler().fit(xs),{k:StandardScaler().fit(v) for k,v in xg.items()},StandardScaler().fit(xa)
def scale(scs,scg,sca,xs,xg,xa): return scs.transform(xs).astype('float32'),{k:scg[k].transform(v).astype('float32') for k,v in xg.items()},sca.transform(xa).astype('float32')
def tt(xs,xg,xa,idx=None):
 if idx is None: return torch.tensor(xs,dtype=torch.float32,device=DEV),{k:torch.tensor(v,dtype=torch.float32,device=DEV) for k,v in xg.items()},torch.tensor(xa,dtype=torch.float32,device=DEV)
 return torch.tensor(xs[idx],dtype=torch.float32,device=DEV),{k:torch.tensor(v[idx],dtype=torch.float32,device=DEV) for k,v in xg.items()},torch.tensor(xa[idx],dtype=torch.float32,device=DEV)
class Model(nn.Module):
 def __init__(self,nnum,gdim,navail,mode):
  super().__init__(); self.names=list(gdim); self.mode=mode; d=HP['d']; drop=HP['dropout']; self.d=d
  self.num=nn.Sequential(nn.Linear(nnum,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,d),nn.LayerNorm(d),nn.GELU())
  self.enc=nn.ModuleDict({k:nn.Sequential(nn.Linear(v,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,d),nn.LayerNorm(d),nn.GELU()) for k,v in gdim.items()})
  self.q=nn.Linear(d,d); self.k=nn.Linear(d,d); self.v=nn.Linear(d,d); self.norm=nn.LayerNorm(d)
  self.av=nn.Sequential(nn.Linear(navail,d//2),nn.LayerNorm(d//2),nn.GELU())
  self.gate=nn.Sequential(nn.Linear(d+d+d//2,d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1))
  self.head=nn.Sequential(nn.Linear(2*d,d),nn.LayerNorm(d),nn.GELU(),nn.Dropout(drop),nn.Linear(d,1)); self.reg=nn.Sequential(nn.Linear(2*d,d//2),nn.GELU(),nn.Linear(d//2,1))
 def forward(self,xs,xg,xa):
  hs=self.num(xs); toks=torch.stack([self.enc[k](xg[k]) for k in self.names],1)
  if 'noattn' in self.mode: w=torch.ones(toks.size(0),toks.size(1),device=toks.device)/toks.size(1); ev=toks.mean(1)
  else:
   score=(self.q(hs).unsqueeze(1)*self.k(toks)).sum(-1)/(self.d**.5); w=torch.softmax(score,1); ev=(w.unsqueeze(-1)*self.v(toks)).sum(1)
  ev=self.norm(ev); av=self.av(xa)
  if 'gate1' in self.mode: g=torch.ones(xs.size(0),device=xs.device)
  elif 'fixedgate' in self.mode: g=torch.sigmoid(xa[:,0]*0.0 + xa.abs().mean(1)*0.35)  # deterministic availability-intensity proxy after scaling
  else: g=torch.sigmoid(self.gate(torch.cat([hs,ev,av],1))).squeeze(1)
  fu=torch.cat([hs,g[:,None]*ev],1); return self.reg(fu).squeeze(1), self.head(fu).squeeze(1), g, w
def train(xs,xg,xa,y,xv,gv,av,yv,mode,seed_value):
 seed(seed_value); m=Model(xs.shape[1],{k:v.shape[1] for k,v in xg.items()},xa.shape[1],mode).to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=HP['lr'],weight_decay=HP['wd'])
 pos=max((y>0).sum(),1); neg=max((y<0).sum(),1); pw=torch.tensor([neg/pos],dtype=torch.float32,device=DEV); best=None; bb=-9; wait=0; n=len(y); XV,GV,AV=tt(xv,gv,av); yvb=(yv>0).astype(int)
 for ep in range(HP['epochs']):
  m.train(); perm=np.random.permutation(n)
  for i in range(0,n,HP['batch']):
   bi=perm[i:i+HP['batch']]; X,G,A=tt(xs,xg,xa,bi); yt=torch.tensor(y[bi],dtype=torch.float32,device=DEV); yb=(yt>0).float(); yr,yd,g,w=m(X,G,A)
   loss=F.binary_cross_entropy_with_logits(yd,yb,pos_weight=pw)+HP['reg_w']*F.smooth_l1_loss(yr,yt)
   if 'learnedgate' in mode: loss=loss+0.001*((g-.55)**2).mean()
   opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),2); opt.step()
  m.eval();
  with torch.no_grad():
   _,yd,g,w=m(XV,GV,AV); pred=(torch.sigmoid(yd).cpu().numpy()>.5).astype(int); b=balanced_accuracy_score(yvb,pred) if len(np.unique(yvb))>1 else 0
  if b>bb: bb=b; wait=0; best={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
  else:
   wait+=1
   if wait>=HP['patience']: break
 if best: m.load_state_dict(best)
 return m,bb
def metrics(y,p,prob,g):
 yt=(y>0).astype(int)
 return dict(n=len(yt),DA=accuracy_score(yt,p),BAcc=balanced_accuracy_score(yt,p),MCC=matthews_corrcoef(yt,p) if len(np.unique(p))>1 else 0,F1=f1_score(yt,p,zero_division=0),pred_pos_rate=float(p.mean()),true_pos_rate=float(yt.mean()),prob_mean=float(prob.mean()),gate_mean=float(g.mean()),gate_std=float(g.std()))
def subsets(df,idx,y,p,prob,g,tr):
 q=df.loc[tr,'qef_global_abs_signal'].quantile(.7); sub=df.loc[idx]; masks={'all':np.ones(len(idx),bool),'event_active':sub.get('qef_global_active_modules',pd.Series(0,index=sub.index)).values>0,'high_signal':sub.get('qef_global_abs_signal',pd.Series(0,index=sub.index)).values>q}
 rows=[]
 for name,mask in masks.items():
  if mask.sum()>=10:
   r=metrics(y[mask],p[mask],prob[mask],g[mask]); r['subset']=name; r['thr']=float(q); rows.append(r)
 return rows
def main():
 df=pd.read_csv(DATA); df.date=pd.to_datetime(df.date); df=df.sort_values('date').reset_index(drop=True)
 num=ex(df,NUM); avail=ex(df,AVAIL); fs=folds(df)
 configs=[
  ('CoreWide_CrossAttn_LearnedGate','core_wide','attn_learnedgate'),
  ('CoreWide_NoAttn_LearnedGate','core_wide','noattn_learnedgate'),
  ('CoreWide_CrossAttn_Gate1','core_wide','attn_gate1'),
  ('CoreWide_CrossAttn_FixedGate','core_wide','attn_fixedgate'),
  ('AvailOnly_CrossAttn_LearnedGate','avail_only','attn_learnedgate'),
  ('FullSignal_CrossAttn_LearnedGate','full_signal','attn_learnedgate'),
]
 rows=[]; print('device',DEV,'folds',[f['fold'] for f in fs], 'seeds', SEEDS, 'horizons', HORIZONS, flush=True)
 for H in HORIZONS:
  target=f'target_H{H}'
  for f in fs:
   tr,va,te=split(df,f)
   for name,var,mode in configs:
    grp=groups(df,var)
    for sd in SEEDS:
     xs,xg,xa,y,itr=build(df,tr,num,grp,avail,target); xv,gv,av,yv,iva=build(df,va,num,grp,avail,target); xt,gt,at,yt,ite=build(df,te,num,grp,avail,target)
     scs,scg,sca=scalefit(xs,xg,xa); xs,xg,xa=scale(scs,scg,sca,xs,xg,xa); xv,gv,av=scale(scs,scg,sca,xv,gv,av); xt,gt,at=scale(scs,scg,sca,xt,gt,at)
     m,vb=train(xs,xg,xa,y,xv,gv,av,yv,mode,sd); m.eval(); X,G,A=tt(xt,gt,at)
     with torch.no_grad(): yr,yd,g,w=m(X,G,A); prob=torch.sigmoid(yd).cpu().numpy(); p=(prob>.5).astype(int); gg=g.cpu().numpy()
     for r in subsets(df,ite,yt,p,prob,gg,tr):
      r.update(dict(H=H,fold=f['fold'],test_year=f['test'],model=name,variant=var,mode=mode,seed=sd,val_BAcc=float(vb),n_num=len(num),n_qef=sum(len(v) for v in grp.values()),n_avail=len(avail))) ; rows.append(r)
     print('done', 'H'+str(H), f['fold'], name, 'seed', sd, 'val',round(vb,4),flush=True)
     del m
     if torch.cuda.is_available(): torch.cuda.empty_cache()
 runs=pd.DataFrame(rows); runs.to_csv(OUT/'h7_ablation_runs.csv',index=False)
 rank=runs.groupby(['model','variant','mode','subset']).agg(DA=('DA','mean'),BAcc=('BAcc','mean'),MCC=('MCC','mean'),F1=('F1','mean'),n=('n','mean'),gate_mean=('gate_mean','mean'),gate_std=('gate_std','mean'),pred_pos_rate=('pred_pos_rate','mean'),runs=('DA','size'),n_qef=('n_qef','mean')).reset_index().sort_values(['subset','MCC','BAcc'],ascending=[True,False,False])
 rank.to_csv(OUT/'h7_ablation_ranking.csv',index=False)
 print('\nSAVED',OUT)
 for subset,sub in rank.groupby('subset'):
  print('\n',subset); print(sub[['model','DA','BAcc','MCC','n','n_qef','gate_mean','gate_std','pred_pos_rate','runs']].to_string(index=False))
if __name__=='__main__': main()
