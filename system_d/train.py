"""Train/evaluate System D with leave-one-session-out CV (speaker-independent).

  python train.py --fusion cross --features mfcc+gfcc --epochs 150
"""
import warnings, os, argparse, time, json; warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np, tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix
from fusion_model import build_fusion

LAB = ("angry", "happy", "neutral", "sad")

p = argparse.ArgumentParser()
p.add_argument("--fusion", default="cross",
               choices=["cross", "self", "concat", "audio_only", "text_only"])
p.add_argument("--features", default="mfcc", choices=["mfcc", "mfcc+gfcc"])
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--folds", default="1,2,3,4,5")
_D = next((q for q in ["iemocap_dataset.npz", "../data/iemocap_dataset.npz",
                        "data/iemocap_dataset.npz"] if __import__("os").path.exists(q)),
          "iemocap_dataset.npz")   # works from the project tree OR the flat Colab bundle
p.add_argument("--data", default=_D)
p.add_argument("--patience", type=int, default=25)
p.add_argument("--out", default="results")
a = p.parse_args()

tf.random.set_seed(a.seed); np.random.seed(a.seed)
z = np.load(a.data, allow_pickle=True)
A = z["mfcc"] if a.features == "mfcc" else np.concatenate([z["mfcc"], z["gfcc"]], -1)
A = A.astype(np.float32)
texts = np.array([str(t) for t in z["transcript"]])
y = np.array([LAB.index(e) for e in z["emotion"]])
ses = z["session"]
print(f"{a.fusion} | {a.features} {A.shape} | seed {a.seed} | "
      f"{'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")

res = []
for f in [int(x) for x in a.folds.split(",")]:
    va_s = (f % 5) + 1                 # a DIFFERENT session for validation
    te = ses == f                      # hold out a whole session = unseen speakers
    va = ses == va_s                   # early stopping must never look at test
    tr = ~te & ~va
    # vocabulary fitted on TRAIN ONLY (no leakage)
    vec = layers.TextVectorization(max_tokens=3000, output_sequence_length=32,
                                   standardize="lower_and_strip_punctuation")
    vec.adapt(texts[tr])
    # per-feature normalisation from train statistics only
    mu, sd = A[tr].mean((0, 1)), A[tr].std((0, 1)) + 1e-8
    Atr, Ate = (A[tr] - mu) / sd, (A[te] - mu) / sd

    m = build_fusion(fusion=a.fusion, audio_dim=A.shape[-1], text_vectorizer=vec)
    m.compile(optimizer=tf.keras.optimizers.legacy.Adam(1e-3, beta_1=0.93, beta_2=0.98),
              loss="categorical_crossentropy", metrics=["accuracy"])
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=a.patience,
                                           restore_best_weights=True, mode="max")]
    t0 = time.time()
    Ava = (A[va] - mu) / sd
    m.fit([Atr, texts[tr]], tf.one_hot(y[tr], 4),
          validation_data=([Ava, texts[va]], tf.one_hot(y[va], 4)),
          epochs=a.epochs, batch_size=a.batch, verbose=0, callbacks=cb)
    pred = m.predict([Ate, texts[te]], verbose=0).argmax(1)
    wa = accuracy_score(y[te], pred); ua = recall_score(y[te], pred, average="macro")
    res.append((wa, ua))
    # keep the best fold's weights so System E has something to demo
    tag = f"{a.fusion}_{a.features.replace('+','')}"
    os.makedirs("models", exist_ok=True)
    best_path = f"models/{tag}_best.json"
    prev = json.load(open(best_path))["wa"] if os.path.exists(best_path) else -1
    if wa > prev:
        m.save_weights(f"models/{tag}_best.weights.h5")
        json.dump({"wa": float(wa), "ua": float(ua), "fold": f,
                   "fusion": a.fusion, "features": a.features,
                   "audio_dim": int(A.shape[-1]),
                   "vocab": vec.get_vocabulary(),
                   "mu": mu.tolist(), "sd": sd.tolist()},
                  open(best_path, "w"))
        print(f"    saved new best model (WA {wa*100:.2f}) -> models/{tag}_best.weights.h5")
    print(f"  fold {f} (test=S{f} n={te.sum():4}, val=S{va_s}, train={tr.sum():4}): "
          f"WA={wa*100:5.2f}  UA={ua*100:5.2f}  [{time.time()-t0:.0f}s]")

wa = np.array([r[0] for r in res]); ua = np.array([r[1] for r in res])
print(f"\n  {a.fusion} / {a.features}:  WA {wa.mean()*100:.2f} +/- {wa.std()*100:.2f}   "
      f"UA {ua.mean()*100:.2f} +/- {ua.std()*100:.2f}")
os.makedirs(a.out, exist_ok=True)
json.dump({"fusion": a.fusion, "features": a.features, "seed": a.seed,
           "wa": wa.tolist(), "ua": ua.tolist(),
           "wa_mean": float(wa.mean()), "ua_mean": float(ua.mean())},
          open(f"{a.out}/{a.fusion}_{a.features.replace('+','')}_s{a.seed}.json", "w"), indent=2)
