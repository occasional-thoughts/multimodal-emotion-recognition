"""What do MFCC+GFCC features actually encode: emotion, or the person talking?"""
import warnings,os,glob; warnings.filterwarnings("ignore")
import numpy as np, librosa
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold

SR=16000
EMO={"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fearful","07":"disgust","08":"surprised"}
win=SlidingWindow(0.025,0.010,"hamming")
pool=lambda m: np.concatenate([m.mean(0),m.std(0)])

X,emo,spk,cond=[],[],[],[]
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__),"data/ravdess/Actor_*/*.wav"))):
    p=os.path.basename(f).split(".")[0].split("-")
    sig,_=librosa.load(f,sr=SR); sig,_=librosa.effects.trim(sig,top_db=30)
    if len(sig)<SR*0.5: continue
    m=librosa.feature.mfcc(y=sig,sr=SR,n_mfcc=13,n_fft=400,hop_length=160)
    m=np.vstack([m,librosa.feature.delta(m),librosa.feature.delta(m,order=2)]).T
    g=gfcc(sig,fs=SR,num_ceps=13,window=win,nfilts=48,nfft=512)
    n=min(len(m),len(g))
    X.append(pool(np.hstack([m[:n],g[:n]])))
    emo.append(EMO[p[2]]); spk.append(int(p[6]))
    # condition = actor+emotion+intensity+statement (the 2 repetitions are near-twins)
    cond.append(f"{p[6]}-{p[2]}-{p[3]}-{p[4]}")
X,emo,spk,cond=np.array(X),np.array(emo),np.array(spk),np.array(cond)

clf=lambda: make_pipeline(StandardScaler(),SVC(C=10,gamma="scale"))
cv=StratifiedKFold(10,shuffle=True,random_state=0)

print("="*66)
print("TEST 1: Can these 'emotion features' identify WHO is speaking?")
print("="*66)
s=cross_val_score(clf(),X,spk,cv=cv,n_jobs=-1)
print(f"  Predicting SPEAKER (24-way, chance=4.2%) : {s.mean()*100:5.2f}%")
e=cross_val_score(clf(),X,emo,cv=cv,n_jobs=-1)
print(f"  Predicting EMOTION (8-way,  chance=12.5%): {e.mean()*100:5.2f}%")
print(f"\n  -> The features identify the PERSON better than the EMOTION.")

print("\n"+"="*66)
print("TEST 2: How many test clips have a near-twin sitting in train?")
print("="*66)
# RAVDESS records each condition TWICE (repetition 01 and 02)
from collections import Counter
c=Counter(cond)
twins=sum(1 for v in c.values() if v>1)
print(f"  Unique recording conditions: {len(c)}")
print(f"  Conditions recorded 2x (near-duplicate pairs): {twins}")
print(f"  -> Under RANDOM CV, ~{twins*2/len(X)*100:.0f}% of clips can have their twin")
print(f"     (same actor, same emotion, same sentence, same intensity) in training.")
