"""Build the aligned IEMOCAP dataset: MFCC (TIM-Net-identical) + GFCC + text + metadata."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, csv, time, librosa
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow

MEAN_LEN, SR, NF = 310000, 22050, 606
win = SlidingWindow(2048 / SR, 512 / SR, "hamming")

def load_sig(path):
    s, fs = librosa.load(path)                      # default sr=22050 as TIM-Net does
    if len(s) < MEAN_LEN:
        p = MEAN_LEN - len(s); r = p % 2; p //= 2
        s = np.pad(s, (p, p + r), "constant", constant_values=0)
    else:
        p = (len(s) - MEAN_LEN) // 2
        s = s[p:p + MEAN_LEN]
    return s, fs

def fit(a, n=NF):
    if len(a) >= n: return a[:n]
    return np.pad(a, ((0, n - len(a)), (0, 0)), mode="edge")

rows = list(csv.DictReader(open("iemocap_index.csv")))
M = np.zeros((len(rows), NF, 39), np.float32)
G = np.zeros((len(rows), NF, 13), np.float32)
t0 = time.time()
for i, r in enumerate(rows):
    s, fs = load_sig(r["wav"])
    M[i] = librosa.feature.mfcc(y=s, sr=fs, n_mfcc=39).T          # identical to TIM-Net
    G[i] = fit(gfcc(s, fs=fs, num_ceps=13, window=win, nfilts=48, nfft=2048))
    if (i + 1) % 500 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(rows)}  {el:.0f}s  eta {el/(i+1)*(len(rows)-i-1):.0f}s", flush=True)

np.savez("iemocap_dataset.npz",
         mfcc=M, gfcc=G,
         utt_id=np.array([r["utt_id"] for r in rows]),
         emotion=np.array([r["emotion"] for r in rows]),
         speaker=np.array([r["speaker"] for r in rows]),
         session=np.array([int(r["session"]) for r in rows]),
         transcript=np.array([r["transcript"] for r in rows], dtype=object))
print(f"\nDONE in {time.time()-t0:.0f}s -> iemocap_dataset.npz")
print(f"  mfcc {M.shape}  gfcc {G.shape}")
