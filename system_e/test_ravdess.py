"""Does the IEMOCAP-trained model work on RAVDESS? (cross-corpus test)"""
import os, glob, warnings, collections; warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from predict import EmotionRecognizer, LAB

R = "/private/tmp/claude-501/-Users-abish-Downloads-Trivia-Duel/e863a175-2fc3-4825-b122-649cad3a4a83/scratchpad/data/ravdess"
EMO = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry"}   # only our 4 classes

rec = EmotionRecognizer()
files = [f for f in sorted(glob.glob(f"{R}/Actor_*/*.wav"))
         if os.path.basename(f).split("-")[2] in EMO]
print(f"model {rec.tag} (trained on IEMOCAP)")
print(f"testing on {len(files)} RAVDESS clips in our 4 classes\n")

import random; random.seed(0); files = random.sample(files, 60)
ok = 0; conf_mat = collections.Counter()
for f in files:
    true = EMO[os.path.basename(f).split("-")[2]]
    # RAVDESS has fixed carrier sentences - give it the correct text, best case for the model
    txt = ("Kids are talking by the door" if os.path.basename(f).split("-")[4] == "01"
           else "Dogs are sitting by the door")
    out = rec.predict(f, transcript=txt)
    ok += out["emotion"] == true
    conf_mat[(true, out["emotion"])] += 1

print(f"cross-corpus accuracy: {ok}/{len(files)} = {ok/len(files)*100:.1f}%")
print(f"(chance = 25%;  same model on held-out IEMOCAP = {rec.meta['wa']*100:.1f}%)\n")
print("what it predicts for each true emotion:")
for t in LAB:
    row = {p: conf_mat[(t, p)] for p in LAB if conf_mat[(t, p)]}
    tot = sum(row.values())
    if tot:
        print(f"  true {t:8} -> " + ", ".join(f"{p} {n}" for p, n in
              sorted(row.items(), key=lambda x: -x[1])))
