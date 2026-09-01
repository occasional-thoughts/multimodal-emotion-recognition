"""Three-way modality comparison: audio-only vs text-only vs fusion on the same clip.

The point: when the streams DISAGREE, you can see the ambiguity the project exists
to resolve - e.g. sarcasm, where the words are positive but the voice is not.
"""
import os, json, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tf_keras; os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
except ImportError:
    pass
import numpy as np, tensorflow as tf
from predict import EmotionRecognizer, extract_features, transcribe, LAB

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "system_d", "models")


def available():
    """Which of the three streams have a trained model on disk?"""
    out = {}
    for tag in os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else []:
        if tag.endswith("_best.json"):
            meta = json.load(open(os.path.join(MODEL_DIR, tag)))
            out[meta["fusion"]] = tag[:-10]
    return out


class ThreeWay:
    """Loads one model per stream: audio_only, text_only, and the best fusion model."""
    def __init__(self):
        av = available()
        fusion_tag = None
        best = -1
        for f in os.listdir(MODEL_DIR):
            if not f.endswith("_best.json"):
                continue
            m = json.load(open(os.path.join(MODEL_DIR, f)))
            if m["fusion"] in ("cross", "self", "concat") and m["wa"] > best:
                best, fusion_tag = m["wa"], f[:-10]
        self.models = {}
        for name, tag in [("audio", av.get("audio_only")),
                          ("text", av.get("text_only")),
                          ("fusion", fusion_tag)]:
            if tag:
                self.models[name] = EmotionRecognizer(tag=tag)
        if not self.models:
            raise FileNotFoundError(f"No models in {MODEL_DIR}")

    def predict(self, wav, transcript=None):
        if transcript is None:
            transcript = transcribe(wav)
        out = {}
        for name, rec in self.models.items():
            r = rec.predict(wav, transcript=transcript)
            out[name] = {"emotion": r["emotion"], "probs": r["probs"],
                         "wa": rec.meta["wa"], "tag": rec.tag}
        first = next(iter(self.models.values()))
        feat, sig, fs = extract_features(wav, first.use_gfcc)
        return {"streams": out, "transcript": transcript, "signal": sig, "sr": fs}


if __name__ == "__main__":
    import csv
    tw = ThreeWay()
    print("loaded streams:", {k: v.tag for k, v in tw.models.items()}, "\n")
    rows = list(csv.DictReader(open("../data/iemocap_index.csv")))
    disagreements = 0
    for r in rows[:12]:
        out = tw.predict(r["wav"], transcript=r["transcript"])
        preds = {k: v["emotion"] for k, v in out["streams"].items()}
        agree = len(set(preds.values())) == 1
        disagreements += not agree
        mark = "" if agree else "  <-- STREAMS DISAGREE"
        print(f"true={r['emotion']:8} " +
              " ".join(f"{k}={v:8}" for k, v in preds.items()) + mark)
    print(f"\n{disagreements}/12 clips had disagreement between streams")
