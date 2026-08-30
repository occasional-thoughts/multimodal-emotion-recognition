"""Regenerate TIM-Net's IEMOCAP MFCC with THEIR exact settings, but keeping utt_ids.
Their get_feature(): librosa.load(default sr=22050) -> pad/crop to 310000 -> mfcc(n_mfcc=39) -> transpose
"""
import numpy as np, csv, sys, librosa
MEAN_LEN = 310000

def get_feature(path, mean_signal_length=MEAN_LEN, embed_len=39):
    signal, fs = librosa.load(path)                 # default sr=22050, exactly as they do
    s_len = len(signal)
    if s_len < mean_signal_length:
        pad = mean_signal_length - s_len
        rem = pad % 2; pad //= 2
        signal = np.pad(signal, (pad, pad + rem), 'constant', constant_values=0)
    else:
        pad = (s_len - mean_signal_length) // 2
        signal = signal[pad:pad + mean_signal_length]
    return np.transpose(librosa.feature.mfcc(y=signal, sr=fs, n_mfcc=embed_len))

rows = list(csv.DictReader(open("iemocap_index.csv")))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
theirs = np.load("TIM-Net_SER/Code/MFCC/IEMOCAP.npy", allow_pickle=True).item()["x"]
print(f"extracting {N} utterances with TIM-Net's exact settings...")

ours, ids = [], []
for r in rows[:N]:
    ours.append(get_feature(r["wav"])); ids.append(r["utt_id"])
ours = np.array(ours, dtype=np.float32)
print(f"our shape {ours.shape}   theirs {theirs.shape}")

# does each of our rows appear somewhere in their matrix?
print("\nmatching our rows against their 5531 rows (nearest neighbour):")
hits = 0
for i in range(min(25, N)):
    d = np.abs(theirs - ours[i]).mean(axis=(1, 2))    # mean abs diff vs every row
    j = int(d.argmin())
    ok = d[j] < 1.0
    hits += ok
    if i < 8:
        print(f"  {ids[i]:26} -> their row {j:5}  diff={d[j]:.4f}  {'MATCH' if ok else 'no match'}")
print(f"\n  {hits}/{min(25,N)} matched their features almost exactly")
print("  -> our extraction reproduces theirs" if hits >= 20 else "  -> settings differ; investigate")
