"""A) Is the twin pair acoustically near-identical? (the leakage MECHANISM)
   B) Same CNN, leaky split vs honest split -> how much is leakage vs architecture?"""
import warnings,os,glob,time,random; warnings.filterwarnings("ignore")
import numpy as np, librosa, torch, torch.nn as nn
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
torch.manual_seed(0); np.random.seed(0); random.seed(0)

SR=16000; MAXF=300
EMO={"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fearful","07":"disgust","08":"surprised"}
LAB=sorted(EMO.values()); L2I={l:i for i,l in enumerate(LAB)}
win=SlidingWindow(0.025,0.010,"hamming")
fix=lambda m: m[:MAXF] if len(m)>=MAXF else np.pad(m,((0,MAXF-len(m)),(0,0)))

seq,vec,y,spk,meta=[],[],[],[],[]
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__),"data/ravdess/Actor_*/*.wav"))):
    p=os.path.basename(f).split(".")[0].split("-")
    sig,_=librosa.load(f,sr=SR); sig,_=librosa.effects.trim(sig,top_db=30)
    if len(sig)<SR*0.5: continue
    m=librosa.feature.mfcc(y=sig,sr=SR,n_mfcc=13,n_fft=400,hop_length=160)
    m=np.vstack([m,librosa.feature.delta(m),librosa.feature.delta(m,order=2)]).T
    g=gfcc(sig,fs=SR,num_ceps=13,window=win,nfilts=48,nfft=512); n=min(len(m),len(g))
    fu=np.hstack([m[:n],g[:n]])
    seq.append(fix(fu)); vec.append(np.concatenate([fu.mean(0),fu.std(0)]))
    y.append(L2I[EMO[p[2]]]); spk.append(int(p[6]))
    meta.append({"actor":p[6],"emo":p[2],"inten":p[3],"stmt":p[4],"rep":p[5]})
seq=np.array(seq,np.float32); vec=np.array(vec,np.float32); y=np.array(y); spk=np.array(spk)

# ---------- A) how similar is a clip to its TWIN vs to other clips? ----------
from scipy.spatial.distance import cosine
V=(vec-vec.mean(0))/(vec.std(0)+1e-8)
idx={}
for i,m in enumerate(meta): idx.setdefault(f"{m['actor']}-{m['emo']}-{m['inten']}-{m['stmt']}",[]).append(i)

def avg(pairs,k=400):
    pairs=random.sample(pairs,min(k,len(pairs)))
    return np.mean([1-cosine(V[a],V[b]) for a,b in pairs])

twins=[(v[0],v[1]) for v in idx.values() if len(v)==2]
same_act_diff=[(i,j) for _ in range(3000) for i,j in [(random.randrange(len(V)),random.randrange(len(V)))]
               if meta[i]['actor']==meta[j]['actor'] and meta[i]['emo']!=meta[j]['emo']]
same_emo_diff_act=[(i,j) for _ in range(3000) for i,j in [(random.randrange(len(V)),random.randrange(len(V)))]
                   if meta[i]['emo']==meta[j]['emo'] and meta[i]['actor']!=meta[j]['actor']]
rnd=[(random.randrange(len(V)),random.randrange(len(V))) for _ in range(3000)]

print("="*64); print("A) COSINE SIMILARITY between clip pairs (1.0 = identical)"); print("="*64)
print(f"  TWIN (same actor+emotion+sentence, take 1 vs take 2) : {avg(twins):.3f}   <-- leaks")
print(f"  same actor, different emotion                        : {avg(same_act_diff):.3f}")
print(f"  same emotion, different actor                        : {avg(same_emo_diff_act):.3f}")
print(f"  random pair                                          : {avg(rnd):.3f}")

# ---------- B) same CNN, leaky split vs honest split ----------
class CNN(nn.Module):
    def __init__(s,d,n=8):
        super().__init__()
        s.c=nn.Sequential(
            nn.Conv1d(d,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.3),
            nn.Conv1d(128,256,5,padding=2),nn.BatchNorm1d(256),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.3),
            nn.Conv1d(256,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU())
        s.h=nn.Linear(128,n)
    def forward(s,x): return s.h(s.c(x).mean(-1))

def train_eval(tr,te,tag):
    mu,sd=seq[tr].mean((0,1)),seq[tr].std((0,1))+1e-8
    X=(seq-mu)/sd
    xtr=torch.tensor(X[tr]).permute(0,2,1); ytr=torch.tensor(y[tr])
    xte=torch.tensor(X[te]).permute(0,2,1); yte=torch.tensor(y[te])
    m=CNN(seq.shape[2]); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    best=0
    for ep in range(60):
        m.train(); perm=torch.randperm(len(xtr))
        for i in range(0,len(xtr),32):
            b=perm[i:i+32]; opt.zero_grad(); lf(m(xtr[b]),ytr[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): best=max(best,(m(xte).argmax(1)==yte).float().mean().item())
    print(f"  {tag:52} {best*100:5.2f}%")
    return best

print("\n"+"="*64); print("B) IDENTICAL CNN + features, only the SPLIT changes"); print("="*64)
n=len(seq); perm=np.random.permutation(n); cut=int(n*0.833)
tr_r=np.zeros(n,bool); tr_r[perm[:cut]]=True          # random split (twins leak)
leaky=train_eval(tr_r,~tr_r,"random split  (twins leak, like many papers)")
hon=train_eval(spk<=20,spk>20,"speaker-independent split (honest)")
print("-"*64); print(f"  gap attributable to the split alone: {(leaky-hon)*100:+.2f} pts")
