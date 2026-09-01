"""Find demo clips the model handles well - restricted to its HELD-OUT session,
so nothing shown was in its training data.

Stage 1: score all held-out clips using gold transcripts (fast).
Stage 2: re-verify the shortlist through the real demo path (Whisper ASR).
"""
import os, csv, json, shutil, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from predict import EmotionRecognizer, extract_features, transcribe, LAB
import tensorflow as tf

rec = EmotionRecognizer()
HELD_OUT = rec.meta["fold"]
print(f"model {rec.tag} | held-out session: S{HELD_OUT} | WA {rec.meta['wa']*100:.2f}\n")

rows = [r for r in csv.DictReader(open("../data/iemocap_index.csv"))
        if int(r["session"]) == HELD_OUT]
# demo-friendly length: long enough to hear, short enough to sit through
rows = [r for r in rows if 1.8 <= float(r["dur"]) <= 7.0 and len(r["transcript"]) > 18]
print(f"{len(rows)} held-out clips in the usable length range\n")

z = np.load("../data/iemocap_dataset.npz", allow_pickle=True)
idx = {u: i for i, u in enumerate(z["utt_id"])}
A = np.concatenate([z["mfcc"], z["gfcc"]], -1).astype(np.float32)
mu, sd = np.array(rec.meta["mu"], np.float32), np.array(rec.meta["sd"], np.float32)

# ---- stage 1: batch-score with gold transcripts ----
ids = [r["utt_id"] for r in rows]
X = (A[[idx[u] for u in ids]] - mu) / sd
txt = tf.constant([r["transcript"] for r in rows])
probs = rec.model.predict([X, txt], verbose=0)

cand = []
for r, pr in zip(rows, probs):
    pred, conf = LAB[int(pr.argmax())], float(pr.max())
    if pred == r["emotion"]:
        cand.append({"utt_id": r["utt_id"], "wav": r["wav"], "true": r["emotion"],
                     "conf": conf, "dur": float(r["dur"]), "text": r["transcript"]})
print(f"{len(cand)} correctly classified with gold transcripts")

# two per emotion, highest confidence first
by = {}
for c in sorted(cand, key=lambda c: -c["conf"]):
    by.setdefault(c["true"], []).append(c)
short = [c for e in LAB for c in by.get(e, [])[:4]]
print(f"shortlist: {len(short)} ({', '.join(f'{e}:{len(by.get(e,[])[:4])}' for e in LAB)})\n")

# ---- stage 2: verify through the actual demo path (Whisper) ----
print("verifying with Whisper ASR (the real demo path)...")
final = []
for c in short:
    out = rec.predict(c["wav"])            # transcript=None -> Whisper
    ok = out["emotion"] == c["true"]
    print(f"  {'OK ' if ok else '-- '} {c['true']:8} conf={max(out['probs'].values()):.2f}  "
          f"asr: {out['transcript'][:44]}")
    if ok:
        c["asr"] = out["transcript"]; c["asr_conf"] = max(out["probs"].values())
        final.append(c)

# keep two per emotion
keep, seen = [], {}
for c in final:
    if seen.get(c["true"], 0) < 2:
        keep.append(c); seen[c["true"]] = seen.get(c["true"], 0) + 1

os.makedirs("demo_clips", exist_ok=True)
manifest = []
for i, c in enumerate(keep, 1):
    name = f"{i:02d}_{c['true']}_{c['utt_id']}.wav"
    shutil.copy(c["wav"], os.path.join("demo_clips", name))
    manifest.append({"file": name, "true_emotion": c["true"], "duration_s": round(c["dur"], 2),
                     "confidence": round(c["asr_conf"], 3), "gold_transcript": c["text"],
                     "whisper_transcript": c["asr"]})
json.dump({"model": rec.tag, "held_out_session": HELD_OUT,
           "note": "All clips are from the model's held-out test session - never seen in training.",
           "clips": manifest}, open("demo_clips/manifest.json", "w"), indent=2)

print(f"\n{len(keep)} clips written to system_e/demo_clips/")
for m in manifest:
    print(f"  {m['file']:44} conf={m['confidence']:.2f}  \"{m['whisper_transcript'][:40]}\"")
