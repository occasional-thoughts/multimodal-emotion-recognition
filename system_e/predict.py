"""System E - inference pipeline. Wraps Systems A-D behind one callable.

Live audio has no transcript, so ASR (Whisper) generates one -> this is exactly
the ASR-robustness condition of Gap 3, and the reason the demo differs from training.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# Only force legacy Keras when the tf_keras shim exists (needed on TF>=2.16 / Colab).
# On TF 2.15 tf.keras is already Keras 2 and setting this breaks the import.
try:
    import tf_keras  # noqa: F401
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
except ImportError:
    pass

import numpy as np, librosa, tensorflow as tf
from tensorflow.keras import layers
from spafe.features.gfcc import gfcc
from spafe.utils.preprocessing import SlidingWindow
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "system_d"))
from fusion_model import build_fusion

LAB = ("angry", "happy", "neutral", "sad")
MEAN_LEN, SR, NF = 310000, 22050, 606
_WIN = SlidingWindow(2048 / SR, 512 / SR, "hamming")
_asr = None


def extract_features(wav_path, use_gfcc=True):
    """Identical preprocessing to training - any mismatch here silently wrecks accuracy."""
    s, fs = librosa.load(wav_path)                      # librosa default sr=22050
    if len(s) < MEAN_LEN:
        p = MEAN_LEN - len(s); r = p % 2; p //= 2
        s = np.pad(s, (p, p + r), "constant", constant_values=0)
    else:
        p = (len(s) - MEAN_LEN) // 2
        s = s[p:p + MEAN_LEN]
    m = librosa.feature.mfcc(y=s, sr=fs, n_mfcc=39).T
    if not use_gfcc:
        return m[:NF], s, fs
    g = gfcc(s, fs=fs, num_ceps=13, window=_WIN, nfilts=48, nfft=2048)
    g = g[:NF] if len(g) >= NF else np.pad(g, ((0, NF - len(g)), (0, 0)), mode="edge")
    return np.hstack([m[:NF], g]), s, fs


def transcribe(wav_path, model_size="base"):
    global _asr
    import whisper
    if _asr is None:
        _asr = whisper.load_model(model_size)
    return _asr.transcribe(wav_path, fp16=False)["text"].strip()


class EmotionRecognizer:
    def __init__(self, model_dir="../system_d/models", tag=None):
        d = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_dir)
        if tag is None:                                  # pick the best available model
            cands = [f for f in os.listdir(d) if f.endswith("_best.json")]
            if not cands:
                raise FileNotFoundError(
                    f"No trained model in {d}. Run system_d/train.py first "
                    "(or copy models/ down from Colab).")
            tag = max(cands, key=lambda f: json.load(open(os.path.join(d, f)))["wa"])[:-10]
        self.meta = json.load(open(os.path.join(d, f"{tag}_best.json")))
        self.tag = tag
        self.use_gfcc = self.meta["features"].endswith("gfcc")
        self.mu = np.array(self.meta["mu"], np.float32)
        self.sd = np.array(self.meta["sd"], np.float32)

        vec = layers.TextVectorization(max_tokens=3000, output_sequence_length=32,
                                       standardize="lower_and_strip_punctuation")
        vec.set_vocabulary(self.meta["vocab"])
        self.model = build_fusion(fusion=self.meta["fusion"],
                                  audio_dim=self.meta["audio_dim"], text_vectorizer=vec)
        self.model.load_weights(os.path.join(d, f"{tag}_best.weights.h5"))

    def predict(self, wav_path, transcript=None):
        feat, sig, fs = extract_features(wav_path, self.use_gfcc)
        if transcript is None:
            transcript = transcribe(wav_path)
        x = ((feat - self.mu) / self.sd)[None].astype(np.float32)
        probs = self.model([x, tf.constant([transcript])], training=False).numpy()[0]
        return {"emotion": LAB[int(probs.argmax())],
                "probs": {l: float(p) for l, p in zip(LAB, probs)},
                "transcript": transcript, "signal": sig, "sr": fs}


if __name__ == "__main__":
    r = EmotionRecognizer()
    print(f"loaded: {r.tag}  (trained WA {r.meta['wa']*100:.2f}, fold {r.meta['fold']}, "
          f"features={r.meta['features']})")
    import csv
    rows = list(csv.DictReader(open("../data/iemocap_index.csv")))
    for row in rows[:3]:
        out = r.predict(row["wav"], transcript=row["transcript"])   # skip ASR for speed
        ok = "OK " if out["emotion"] == row["emotion"] else "-- "
        print(f"  {ok}true={row['emotion']:8} pred={out['emotion']:8} "
              f"| {row['transcript'][:40]}")
