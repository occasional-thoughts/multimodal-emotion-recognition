"""Does the RAVDESS leakage finding transfer to IEMOCAP? (spontaneous, no scripted twins)"""
import warnings,os,csv,time; warnings.filterwarnings("ignore")
import numpy as np, librosa
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold, LeaveOneGroupOut

SR=16000; win=SlidingWindow(0.025,0.010,"hamming")
D=os.path.dirname(os.path.abspath(__file__)); CACHE=os.path.join(D,"iemocap_feats.npz")

if os.path.exists(CACHE):
    z=np.load(CACHE,allow_pickle=True); Xm,Xb,y,spk,ses=z["Xm"],z["Xb"],z["y"],z["spk"],z["ses"]
else:
    rows=list(csv.DictReader(open(os.path.join(D,"iemocap_index.csv"))))
    Xm,Xb,y,spk,ses=[],[],[],[],[]; t0=time.time()
    for i,r in enumerate(rows):
        try:
            sig,_=librosa.load(r["wav"],sr=SR); sig,_=librosa.effects.trim(sig,top_db=30)
            if len(sig)<SR*0.4: continue
            m=librosa.feature.mfcc(y=sig,sr=SR,n_mfcc=13,n_fft=400,hop_length=160)
            m=np.vstack([m,librosa.feature.delta(m),librosa.feature.delta(m,order=2)]).T
            g=gfcc(sig,fs=SR,num_ceps=13,window=win,nfilts=48,nfft=512); n=min(len(m),len(g))
            p=lambda a: np.concatenate([a.mean(0),a.std(0)])
            Xm.append(p(m[:n])); Xb.append(p(np.hstack([m[:n],g[:n]])))
            y.append(r["emotion"]); spk.append(r["speaker"]); ses.append(int(r["session"]))
        except Exception: pass
        if (i+1)%1500==0: print(f"  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)")
    Xm,Xb,y,spk,ses=map(np.array,(Xm,Xb,y,spk,ses))
    np.savez(CACHE,Xm=Xm,Xb=Xb,y=y,spk=spk,ses=ses)
    print(f"features extracted in {time.time()-t0:.0f}s")

print(f"\n{Xm.shape[0]} utts | {len(set(spk))} speakers | {len(set(ses))} sessions | classes {sorted(set(y))}\n")
clf=lambda: make_pipeline(StandardScaler(),SVC(C=10,gamma="scale"))

print("="*68); print("IEMOCAP: same features + classifier, only the SPLIT changes"); print("="*68)
a=cross_val_score(clf(),Xb,y,cv=StratifiedKFold(10,shuffle=True,random_state=0),n_jobs=-1)
b=cross_val_score(clf(),Xb,y,cv=GroupKFold(10),groups=spk,n_jobs=-1)
c=cross_val_score(clf(),Xb,y,cv=LeaveOneGroupOut(),groups=ses,n_jobs=-1)
print(f"  1. random 10-fold CV      (speakers leak)        : {a.mean()*100:5.2f}%  ±{a.std()*100:.2f}")
print(f"  2. speaker-independent CV (no speaker overlap)   : {b.mean()*100:5.2f}%  ±{b.std()*100:.2f}")
print(f"  3. leave-one-session-out  (IEMOCAP standard)     : {c.mean()*100:5.2f}%  ±{c.std()*100:.2f}")
print("-"*68)
print(f"  inflation from speaker leakage: {(a.mean()-b.mean())*100:+.2f} pts")

print("\n"+"="*68); print("Does GFCC help on IEMOCAP? (speaker-independent)"); print("="*68)
m1=cross_val_score(clf(),Xm,y,cv=GroupKFold(10),groups=spk,n_jobs=-1).mean()
print(f"  MFCC only  : {m1*100:5.2f}%")
print(f"  MFCC+GFCC  : {b.mean()*100:5.2f}%     -> GFCC {(b.mean()-m1)*100:+.2f} pts")
