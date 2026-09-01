"""Find held-out clips where AUDIO is right, TEXT is wrong, and FUSION is right.
The mirror of the usual case - shows the acoustic stream carrying the decision.
"""
import os, csv, json, shutil, warnings; warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, tensorflow as tf
from compare import ThreeWay
from predict import LAB

tw = ThreeWay()
HELD = tw.models["fusion"].meta["fold"]
print(f"searching held-out session S{HELD}\n")

rows = [r for r in csv.DictReader(open("../data/iemocap_index.csv"))
        if int(r["session"]) == HELD and 1.8 <= float(r["dur"]) <= 7.0
        and len(r["transcript"]) > 18]

z = np.load("../data/iemocap_dataset.npz", allow_pickle=True)
idx = {u: i for i, u in enumerate(z["utt_id"])}
MF = z["mfcc"].astype(np.float32)
BOTH = np.concatenate([z["mfcc"], z["gfcc"]], -1).astype(np.float32)
ids = [r["utt_id"] for r in rows]
txt = tf.constant([r["transcript"] for r in rows])

def score(rec, feats):
    mu = np.array(rec.meta["mu"], np.float32); sd = np.array(rec.meta["sd"], np.float32)
    X = (feats[[idx[u] for u in ids]] - mu) / sd
    return rec.model.predict([X, txt], verbose=0)

pa = score(tw.models["audio"], MF)
pt = score(tw.models["text"], MF)
pf = score(tw.models["fusion"], BOTH)

hits = []
for r, a, t, f in zip(rows, pa, pt, pf):
    A, T, F = LAB[a.argmax()], LAB[t.argmax()], LAB[f.argmax()]
    if A == r["emotion"] and T != r["emotion"] and F == r["emotion"]:
        hits.append({**r, "a_conf": float(a.max()), "t_wrong": T, "f_conf": float(f.max())})
print(f"{len(hits)} clips where audio is right, text is wrong, fusion is right\n")

hits.sort(key=lambda h: -(h["a_conf"] + h["f_conf"]))
print("verifying top candidates through Whisper (the real demo path)...")
chosen = None
for h in hits[:8]:
    out = tw.predict(h["wav"])                      # Whisper ASR
    s = out["streams"]
    A, T, F = s["audio"]["emotion"], s["text"]["emotion"], s["fusion"]["emotion"]
    ok = A == h["emotion"] and T != h["emotion"] and F == h["emotion"]
    print(f"  {'OK ' if ok else '-- '} true={h['emotion']:8} audio={A:8} text={T:8} fusion={F:8}"
          f"  \"{out['transcript'][:38]}\"")
    if ok and chosen is None:
        chosen = (h, out)

if chosen:
    h, out = chosen
    name = f"09_{h['emotion']}_AUDIOWINS_{h['utt_id']}.wav"
    shutil.copy(h["wav"], os.path.join("demo_clips", name))
    print(f"\nsaved: demo_clips/{name}")
    print(f"  true      : {h['emotion']}")
    for k in ["audio", "text", "fusion"]:
        p = out["streams"][k]["probs"]; top = max(p, key=p.get)
        print(f"  {k:9} : {top:8} ({p[top]:.3f})")
    print(f"  transcript: {out['transcript'][:60]}")
    m = json.load(open("demo_clips/manifest.json"))
    m["clips"].append({"file": name, "true_emotion": h["emotion"],
        "duration_s": round(float(h["dur"]), 2), "note": "audio correct, text wrong, fusion correct",
        "gold_transcript": h["transcript"], "whisper_transcript": out["transcript"]})
    json.dump(m, open("demo_clips/manifest.json", "w"), indent=2)
else:
    print("\nno candidate survived Whisper verification")
