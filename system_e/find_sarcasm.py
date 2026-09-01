"""Sarcasm signature: text model reads POSITIVE, the utterance is actually NEGATIVE,
and fusion recovers the true label. Positive words, negative delivery.
"""
import os, csv, json, shutil, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, tensorflow as tf
from compare import ThreeWay
from predict import LAB

tw = ThreeWay(); HELD = tw.models["fusion"].meta["fold"]
rows = [r for r in csv.DictReader(open("../data/iemocap_index.csv"))
        if int(r["session"]) == HELD and 1.5 <= float(r["dur"]) <= 8.0
        and len(r["transcript"]) > 14]
z = np.load("../data/iemocap_dataset.npz", allow_pickle=True)
idx = {u: i for i, u in enumerate(z["utt_id"])}
MF = z["mfcc"].astype(np.float32); BOTH = np.concatenate([z["mfcc"], z["gfcc"]], -1).astype(np.float32)
ids = [r["utt_id"] for r in rows]; txt = tf.constant([r["transcript"] for r in rows])

def score(rec, feats):
    mu = np.array(rec.meta["mu"], np.float32); sd = np.array(rec.meta["sd"], np.float32)
    return rec.model.predict([(feats[[idx[u] for u in ids]] - mu) / sd, txt], verbose=0)

pt = score(tw.models["text"], MF); pf = score(tw.models["fusion"], BOTH)

NEG = {"angry", "sad"}
hits = []
for r, t, f in zip(rows, pt, pf):
    T, F = LAB[t.argmax()], LAB[f.argmax()]
    if r["emotion"] in NEG and T == "happy" and F == r["emotion"]:
        hits.append({**r, "t_conf": float(t.max()), "f_conf": float(f.max())})
print(f"{len(hits)} clips: text says HAPPY, truth is negative, fusion correct\n")

hits.sort(key=lambda h: -(h["t_conf"] + h["f_conf"]))
for h in hits[:10]:
    print(f"  true={h['emotion']:6} text_conf={h['t_conf']:.2f} fusion_conf={h['f_conf']:.2f}  \"{h['transcript'][:52]}\"")

print("\nverifying through Whisper...")
chosen = None
for h in hits[:8]:
    out = tw.predict(h["wav"]); s = out["streams"]
    T, F, A = s["text"]["emotion"], s["fusion"]["emotion"], s["audio"]["emotion"]
    ok = T == "happy" and F == h["emotion"]
    print(f"  {'OK ' if ok else '-- '} true={h['emotion']:6} audio={A:8} text={T:8} fusion={F:8}  \"{out['transcript'][:36]}\"")
    if ok and chosen is None: chosen = (h, out)

if chosen:
    h, out = chosen
    name = f"10_{h['emotion']}_SARCASM_{h['utt_id']}.wav"
    shutil.copy(h["wav"], os.path.join("demo_clips", name))
    print(f"\nsaved: demo_clips/{name}")
    print(f"  true      : {h['emotion']}")
    for k in ["audio", "text", "fusion"]:
        p = out["streams"][k]["probs"]; top = max(p, key=p.get)
        print(f"  {k:9} : {top:8} ({p[top]:.3f})")
    print(f"  transcript: \"{out['transcript'][:66]}\"")
    m = json.load(open("demo_clips/manifest.json"))
    m["clips"].append({"file": name, "true_emotion": h["emotion"],
        "duration_s": round(float(h["dur"]), 2),
        "note": "sarcasm case: text reads positive, truth is negative, fusion correct",
        "gold_transcript": h["transcript"], "whisper_transcript": out["transcript"]})
    json.dump(m, open("demo_clips/manifest.json", "w"), indent=2)
else:
    print("\nnone survived verification")
