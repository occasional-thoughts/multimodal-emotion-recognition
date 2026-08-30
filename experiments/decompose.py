"""Decompose the 20-point gap into its two causes."""
import warnings,os,glob; warnings.filterwarnings("ignore")
import numpy as np, librosa
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold

SR=16000
EMO={"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fearful","07":"disgust","08":"surprised"}
win=SlidingWindow(0.025,0.010,"hamming"); pool=lambda m: np.concatenate([m.mean(0),m.std(0)])
X,emo,spk,cond=[],[],[],[]
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__),"data/ravdess/Actor_*/*.wav"))):
    p=os.path.basename(f).split(".")[0].split("-")
    sig,_=librosa.load(f,sr=SR); sig,_=librosa.effects.trim(sig,top_db=30)
    if len(sig)<SR*0.5: continue
    m=librosa.feature.mfcc(y=sig,sr=SR,n_mfcc=13,n_fft=400,hop_length=160)
    m=np.vstack([m,librosa.feature.delta(m),librosa.feature.delta(m,order=2)]).T
    g=gfcc(sig,fs=SR,num_ceps=13,window=win,nfilts=48,nfft=512); n=min(len(m),len(g))
    X.append(pool(np.hstack([m[:n],g[:n]]))); emo.append(EMO[p[2]]); spk.append(int(p[6]))
    cond.append(f"{p[6]}-{p[2]}-{p[3]}-{p[4]}")
X,emo,spk,cond=np.array(X),np.array(emo),np.array(spk),np.array(cond)
clf=lambda: make_pipeline(StandardScaler(),SVC(C=10,gamma="scale"))

a=cross_val_score(clf(),X,emo,cv=StratifiedKFold(10,shuffle=True,random_state=0),n_jobs=-1).mean()
b=cross_val_score(clf(),X,emo,cv=GroupKFold(10),groups=cond,n_jobs=-1).mean()
c=cross_val_score(clf(),X,emo,cv=GroupKFold(10),groups=spk,n_jobs=-1).mean()

print(f"{'1. Random 10-fold CV  (twins leak + speaker leaks)':<52} {a*100:5.2f}%")
print(f"{'2. Group by condition (twins blocked, speaker leaks)':<52} {b*100:5.2f}%")
print(f"{'3. Group by speaker   (both blocked = honest)':<52} {c*100:5.2f}%")
print("-"*60)
print(f"   cost of blocking near-duplicate twins : {(a-b)*100:5.2f} pts")
print(f"   cost of blocking speaker identity     : {(b-c)*100:5.2f} pts")
print(f"   TOTAL inflation                       : {(a-c)*100:5.2f} pts")
