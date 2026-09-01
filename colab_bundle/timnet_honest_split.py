"""How much of TIM-Net's reported 71.65% on IEMOCAP survives an honest split?

TIM-Net reports 71.65% using 10-fold RANDOM cross-validation, which lets the same
speaker appear in train and test. This runs TIM-Net's EXACT architecture and
hyperparameters, changing ONLY the split: leave-one-session-out (speaker-independent).

The difference isolates evaluation-protocol inflation from architecture.

  python timnet_honest_split.py --mode session   # honest
  python timnet_honest_split.py --mode random    # reproduces their protocol
"""
import warnings, os, argparse, json, time; warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
try:
    import tf_keras; os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
except ImportError:
    pass

import numpy as np, tensorflow as tf, sys
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score

from TIMNET import TIMNET
from Model import WeightLayer, smooth_labels
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

LAB = ("angry", "happy", "neutral", "sad")

p = argparse.ArgumentParser()
p.add_argument("--mode", default="session", choices=["session", "random"])
p.add_argument("--epochs", type=int, default=300)   # TIM-Net uses 500; 300 + early stop is plenty
_D = next((q for q in ["../data/iemocap_dataset.npz", "data/iemocap_dataset.npz",
                        "iemocap_dataset.npz"] if os.path.exists(q)),
          "../data/iemocap_dataset.npz")
p.add_argument("--data", default=_D)
p.add_argument("--seed", type=int, default=16)      # TIM-Net's IEMOCAP seed
a = p.parse_args()
tf.random.set_seed(a.seed); np.random.seed(a.seed)

z = np.load(a.data, allow_pickle=True)
X = z["mfcc"].astype(np.float32)                    # identical to TIM-Net's own features
y = np.array([LAB.index(e) for e in z["emotion"]])
ses = z["session"]
Y = tf.keras.utils.to_categorical(y, 4)
print(f"mode={a.mode} | X{X.shape} | {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")


def build():
    """TIM-Net exactly as published: dilations=10 for IEMOCAP, 39 filters, kernel 2."""
    inp = Input(shape=X.shape[1:])
    md = TIMNET(nb_filters=39, kernel_size=2, nb_stacks=1, dilations=10,
                dropout_rate=0.1, activation="relu", name="TIMNET")(inp)
    out = Dense(4, activation="softmax")(WeightLayer()(md))
    m = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.001, beta_1=0.93, beta_2=0.98),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m


# build the folds
if a.mode == "session":
    folds = [(ses != s, ses == s) for s in sorted(set(ses))]      # leave-one-session-out
else:
    skf = StratifiedKFold(10, shuffle=True, random_state=a.seed)  # TIM-Net's protocol
    folds = [(np.isin(np.arange(len(y)), tr), np.isin(np.arange(len(y)), te))
             for tr, te in skf.split(X, y)]

res = []
for i, (tr, te) in enumerate(folds, 1):
    m = build()
    t0 = time.time()
    m.fit(X[tr], smooth_labels(Y[tr].copy(), 0.1),      # TIM-Net uses label smoothing 0.1
          batch_size=64, epochs=a.epochs, verbose=0,
          validation_data=(X[te], Y[te]),
          callbacks=[tf.keras.callbacks.EarlyStopping(
              monitor="val_accuracy", patience=30, restore_best_weights=True, mode="max")])
    pred = m.predict(X[te], verbose=0).argmax(1)
    wa = accuracy_score(y[te], pred); ua = recall_score(y[te], pred, average="macro")
    res.append((wa, ua))
    print(f"  fold {i} (n={te.sum():5}): WA={wa*100:5.2f}  UA={ua*100:5.2f}  [{time.time()-t0:.0f}s]",
          flush=True)

wa = np.array([r[0] for r in res]); ua = np.array([r[1] for r in res])
print(f"\n  TIM-Net / {a.mode} split:  WA {wa.mean()*100:.2f} +/- {wa.std()*100:.2f}   "
      f"UA {ua.mean()*100:.2f} +/- {ua.std()*100:.2f}")
print(f"  TIM-Net published (10-fold random CV): 71.65")
print(f"  difference: {wa.mean()*100 - 71.65:+.2f} points")
os.makedirs("results_timnet", exist_ok=True)
json.dump({"mode": a.mode, "wa": wa.tolist(), "ua": ua.tolist(),
           "wa_mean": float(wa.mean()), "ua_mean": float(ua.mean()),
           "published_random_cv": 71.65},
          open(f"results_timnet/timnet_{a.mode}.json", "w"), indent=2)
