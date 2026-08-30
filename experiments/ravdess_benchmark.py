"""Real benchmark on RAVDESS: does MFCC+GFCC actually beat MFCC alone?
Speaker-independent split (no actor appears in both train and test).
"""
import warnings, time, os, glob, sys
warnings.filterwarnings("ignore")
import numpy as np
import librosa
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix

SR = 16000
EMO = {"01":"neutral","02":"calm","03":"happy","04":"sad",
       "05":"angry","06":"fearful","07":"disgust","08":"surprised"}

files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "data/ravdess/Actor_*/*.wav")))
print(f"found {len(files)} wav files\n")

win = SlidingWindow(0.025, 0.010, "hamming")

def pool(m):
    """(frames, d) -> (2d,) mean+std pooling"""
    return np.concatenate([m.mean(0), m.std(0)])

X_mfcc, X_both, y, spk = [], [], [], []
t0 = time.time()
for i, f in enumerate(files):
    parts = os.path.basename(f).split(".")[0].split("-")
    emo, actor = EMO[parts[2]], int(parts[6])
    try:
        sig, _ = librosa.load(f, sr=SR)
        sig, _ = librosa.effects.trim(sig, top_db=30)          # trim silence
        if len(sig) < SR * 0.5:
            continue
        m = librosa.feature.mfcc(y=sig, sr=SR, n_mfcc=13, n_fft=400, hop_length=160)
        m = np.vstack([m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)]).T
        g = gfcc(sig, fs=SR, num_ceps=13, window=win, nfilts=48, nfft=512)
        n = min(len(m), len(g))
        X_mfcc.append(pool(m[:n]))
        X_both.append(pool(np.hstack([m[:n], g[:n]])))
        y.append(emo); spk.append(actor)
    except Exception as e:
        print(f"  skip {os.path.basename(f)}: {e}")
    if (i + 1) % 300 == 0:
        print(f"  {i+1}/{len(files)} processed ({time.time()-t0:.0f}s)")

X_mfcc = np.array(X_mfcc); X_both = np.array(X_both)
y = np.array(y); spk = np.array(spk)
print(f"\nextracted in {time.time()-t0:.0f}s")
print(f"MFCC only    feature matrix {X_mfcc.shape}")
print(f"MFCC+GFCC    feature matrix {X_both.shape}")
print(f"classes: {sorted(set(y))}\n")

# speaker-independent split: actors 1-20 train, 21-24 test
tr, te = spk <= 20, spk > 20
print(f"train {tr.sum()} utts (actors 1-20)   test {te.sum()} utts (actors 21-24)")
print("=" * 62)

results = {}
for name, X in [("MFCC only (78-d)", X_mfcc), ("MFCC+GFCC (104-d)", X_both)]:
    clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", kernel="rbf"))
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    wa = accuracy_score(y[te], pred)                       # weighted accuracy
    ua = recall_score(y[te], pred, average="macro")        # unweighted (class-balanced)
    results[name] = (wa, ua, pred)
    print(f"{name:20}  WA={wa*100:5.2f}%   UA={ua*100:5.2f}%")

d_wa = (results["MFCC+GFCC (104-d)"][0] - results["MFCC only (78-d)"][0]) * 100
d_ua = (results["MFCC+GFCC (104-d)"][1] - results["MFCC only (78-d)"][1]) * 100
print("=" * 62)
print(f"GFCC contribution:    WA {d_wa:+.2f} pts    UA {d_ua:+.2f} pts")

labels = sorted(set(y))
print(f"\nconfusion matrix (MFCC+GFCC), rows=true cols=pred")
print("            " + " ".join(f"{l[:4]:>5}" for l in labels))
cm = confusion_matrix(y[te], results["MFCC+GFCC (104-d)"][2], labels=labels)
for l, row in zip(labels, cm):
    print(f"{l:>11} " + " ".join(f"{v:5d}" for v in row))
