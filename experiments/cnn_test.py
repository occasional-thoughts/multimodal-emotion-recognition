"""GFCC contribution under a TEMPORAL model (1D CNN) instead of mean+std pooling.
Speaker-independent split throughout."""
import warnings,os,glob,time; warnings.filterwarnings("ignore")
import numpy as np, librosa, torch, torch.nn as nn
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
torch.manual_seed(0); np.random.seed(0)

SR=16000; MAXF=300
EMO={"01":"neutral","02":"calm","03":"happy","04":"sad","05":"angry","06":"fearful","07":"disgust","08":"surprised"}
LAB=sorted(EMO.values()); L2I={l:i for i,l in enumerate(LAB)}
win=SlidingWindow(0.025,0.010,"hamming")

def fix(m):
    if len(m)>=MAXF: return m[:MAXF]
    return np.pad(m,((0,MAXF-len(m)),(0,0)))

M,G,y,spk=[],[],[],[]
for f in sorted(glob.glob(os.path.join(os.path.dirname(__file__),"data/ravdess/Actor_*/*.wav"))):
    p=os.path.basename(f).split(".")[0].split("-")
    sig,_=librosa.load(f,sr=SR); sig,_=librosa.effects.trim(sig,top_db=30)
    if len(sig)<SR*0.5: continue
    m=librosa.feature.mfcc(y=sig,sr=SR,n_mfcc=13,n_fft=400,hop_length=160)
    m=np.vstack([m,librosa.feature.delta(m),librosa.feature.delta(m,order=2)]).T
    g=gfcc(sig,fs=SR,num_ceps=13,window=win,nfilts=48,nfft=512)
    n=min(len(m),len(g))
    M.append(fix(m[:n])); G.append(fix(g[:n])); y.append(L2I[EMO[p[2]]]); spk.append(int(p[6]))
M,G,y,spk=np.array(M,np.float32),np.array(G,np.float32),np.array(y),np.array(spk)
print(f"MFCC {M.shape}  GFCC {G.shape}  labels {y.shape}\n")

class CNN(nn.Module):
    def __init__(s,d,n=8):
        super().__init__()
        s.c=nn.Sequential(
            nn.Conv1d(d,128,5,padding=2),nn.BatchNorm1d(128),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.3),
            nn.Conv1d(128,256,5,padding=2),nn.BatchNorm1d(256),nn.ReLU(),nn.MaxPool1d(2),nn.Dropout(0.3),
            nn.Conv1d(256,128,3,padding=1),nn.BatchNorm1d(128),nn.ReLU())
        s.h=nn.Linear(128,n)
    def forward(s,x): return s.h(s.c(x).mean(-1))

def run(X,tag):
    tr,te=spk<=20,spk>20
    # per-feature normalization from train stats
    mu,sd=X[tr].mean((0,1)),X[tr].std((0,1))+1e-8
    Xn=(X-mu)/sd
    xtr=torch.tensor(Xn[tr]).permute(0,2,1); ytr=torch.tensor(y[tr])
    xte=torch.tensor(Xn[te]).permute(0,2,1); yte=torch.tensor(y[te])
    m=CNN(X.shape[2]); opt=torch.optim.Adam(m.parameters(),1e-3,weight_decay=1e-4)
    lf=nn.CrossEntropyLoss(); best=0
    for ep in range(60):
        m.train(); perm=torch.randperm(len(xtr))
        for i in range(0,len(xtr),32):
            b=perm[i:i+32]; opt.zero_grad()
            lf(m(xtr[b]),ytr[b]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            acc=(m(xte).argmax(1)==yte).float().mean().item()
        best=max(best,acc)
    print(f"{tag:24} best test WA = {best*100:5.2f}%")
    return best

t0=time.time()
a=run(M,"MFCC only (39-d)")
b=run(np.concatenate([M,G],2),"MFCC+GFCC (52-d)")
print(f"\n>>> GFCC contribution with temporal CNN: {(b-a)*100:+.2f} pts   ({time.time()-t0:.0f}s)")
