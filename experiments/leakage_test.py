"""Same features, same classifier — only the SPLIT changes.
Tests whether reported SER accuracies are inflated by speaker leakage."""
import warnings, os, glob; warnings.filterwarnings("ignore")
import numpy as np, librosa
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold

SR=16000
EMO={"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fearful","07":"disgust","08":"surprised"}
win=SlidingWindow(0.025,0.010,"hamming")
pool=lambda m: np.concatenate([m.mean(0), m.std(0)])

X,y,spk=[],[],[]
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__),"data/ravdess/Actor_*/*.wav"))):
    p=os.path.basename(f).split(".")[0].split("-")
    sig,_=librosa.load(f,sr=SR); sig,_=librosa.effects.trim(sig,top_db=30)
    if len(sig)<SR*0.5: continue
    m=librosa.feature.mfcc(y=sig,sr=SR,n_mfcc=13,n_fft=400,hop_length=160)
    m=np.vstack([m,librosa.feature.delta(m),librosa.feature.delta(m,order=2)]).T
    g=gfcc(sig,fs=SR,num_ceps=13,window=win,nfilts=48,nfft=512)
    n=min(len(m),len(g))
    X.append(pool(np.hstack([m[:n],g[:n]]))); y.append(EMO[p[2]]); spk.append(int(p[6]))
X,y,spk=np.array(X),np.array(y),np.array(spk)
print(f"{X.shape[0]} utterances, {len(set(spk))} speakers, {len(set(y))} classes\n")

clf=lambda: make_pipeline(StandardScaler(), SVC(C=10,gamma="scale"))

# A) random 10-fold CV — speakers appear in BOTH train and test (what many papers do)
a=cross_val_score(clf(),X,y,cv=StratifiedKFold(10,shuffle=True,random_state=0),n_jobs=-1)
# B) speaker-independent group CV — no speaker overlap
b=cross_val_score(clf(),X,y,cv=GroupKFold(10),groups=spk,n_jobs=-1)

print(f"A) random 10-fold CV      (speakers LEAK across folds): {a.mean()*100:5.2f}%  ±{a.std()*100:.2f}")
print(f"B) speaker-independent CV (no speaker overlap)        : {b.mean()*100:5.2f}%  ±{b.std()*100:.2f}")
print(f"\n>>> INFLATION FROM SPEAKER LEAKAGE: {(a.mean()-b.mean())*100:+.2f} percentage points <<<")
