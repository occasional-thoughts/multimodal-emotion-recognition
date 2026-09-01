"""Trace one utterance through System B, printing the shape at every stage."""
import warnings, os, csv; warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, librosa
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow

r = [x for x in csv.DictReader(open("../data/iemocap_index.csv"))][3]
print(f"utterance : {r['utt_id']}   emotion={r['emotion']}   \"{r['transcript'][:44]}\"\n")

# ---- 1. load ----
sig, fs = librosa.load(r["wav"])
print(f"1  LOAD          in : {r['utt_id']}.wav")
print(f"                out: {sig.shape} float32 @ {fs} Hz  = {len(sig)/fs:.2f} s")

# ---- 2. standardise length ----
MEAN = 310000
raw = len(sig)
if raw < MEAN:
    p = MEAN - raw; rem = p % 2; p //= 2
    sig = np.pad(sig, (p, p + rem), "constant")
    how = f"zero-padded {p} samples each side"
else:
    p = (raw - MEAN)//2; sig = sig[p:p+MEAN]; how = f"centre-cropped, dropped {raw-MEAN}"
print(f"\n2  STANDARDISE   in : ({raw},)")
print(f"                out: {sig.shape}  ({how})  = {len(sig)/fs:.2f} s exactly")

# ---- 3a. MFCC ----
n_fft, hop = 2048, 512
S = np.abs(librosa.stft(sig, n_fft=n_fft, hop_length=hop))
print(f"\n3a MFCC")
print(f"   framing       in : {sig.shape}")
print(f"                out: {S.shape}  = ({n_fft//2+1} freq bins, {S.shape[1]} frames)")
mel = librosa.feature.melspectrogram(S=S**2, sr=fs, n_mels=128)
print(f"   mel filterbank in: {S.shape}   -> 128 triangular filters on the mel scale")
print(f"                out: {mel.shape}")
logmel = librosa.power_to_db(mel)
print(f"   log            in: {mel.shape}  -> compress dynamic range")
print(f"                out: {logmel.shape}")
m = librosa.feature.mfcc(y=sig, sr=fs, n_mfcc=39).T
print(f"   DCT, keep 39   in: {logmel.shape}  -> decorrelate, keep low quefrency")
print(f"                out: {m.shape}  = (frames, 39)   <-- MFCC")

# ---- 3b. GFCC ----
win = SlidingWindow(n_fft/fs, hop/fs, "hamming")
g = gfcc(sig, fs=fs, num_ceps=13, window=win, nfilts=48, nfft=n_fft)
print(f"\n3b GFCC")
print(f"   same framing   in: {sig.shape}   -> 48 gammatone filters on the ERB scale")
print(f"   DCT, keep 13  out: {g.shape}  = (frames, 13)   <-- GFCC")
g = g[:606] if len(g) >= 606 else np.pad(g, ((0, 606-len(g)), (0, 0)), mode="edge")
print(f"   pad to match  out: {g.shape}")

# ---- 4. concatenate ----
X = np.hstack([m[:606], g])
print(f"\n4  CONCATENATE   in : MFCC {m[:606].shape} + GFCC {g.shape}")
print(f"                out: {X.shape}   <-- what System B receives")

# ---- 5. through TIM-Net ----
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import sys; sys.path.insert(0, "../baselines/TIM-Net_SER/Code")
from TIMNET import TIMNET
from Model import WeightLayer

inp = Input(shape=(606, 52))
md = TIMNET(nb_filters=39, kernel_size=2, nb_stacks=1, dilations=10,
            dropout_rate=0.1, activation="relu", name="TIMNET")(inp)
dec = WeightLayer()(md)
out = Dense(4, activation="softmax")(dec)
mdl = Model(inp, [md, dec, out])
a, b, c = mdl(X[None].astype("float32"))
print(f"\n5  TIM-NET       in : (1, 606, 52)")
print(f"                out: {tuple(a.shape)}  = one 39-d vector per dilation level")
print(f"\n6  WEIGHTLAYER   in : {tuple(a.shape)}  -> learned weighted sum over 10 scales")
print(f"                out: {tuple(b.shape)}   <-- the utterance embedding")
print(f"\n7  CLASSIFIER    in : {tuple(b.shape)}")
print(f"                out: {tuple(c.shape)}    = P(angry), P(happy), P(neutral), P(sad)")
