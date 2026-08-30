# Team Work Split

Status: Systems A-E built. Baselines done. Fusion experiments running.

| | WA | UA |
|---|---|---|
| audio only (System B) | 53.86 +/- 1.62 | 55.56 +/- 1.59 |
| text only (System C) | 60.15 +/- 2.89 | 61.15 +/- 3.17 |
| concat (no attention) | running | |
| cross / self attention | running | |

---

## The big win: 4 more free GPUs

Colab's free T4 quota is **per Google account**. Each person running experiments
on their own account multiplies our throughput ~4x. Nobody needs to wait.

**Everyone: use your OWN Google account.** Don't share one login - that shares one quota.

---

## Assignments

### Person 1 (Abisha) - owner
Main experiment matrix: `cross`/`self` x `mfcc`/`mfcc+gfcc`, seed 0.
Keeps `results/` and `models/` as the master copy.

### Person 2 - seed replication (needs data)
Run seeds 1 and 2 so we can report mean +/- std and run a t-test.
Single seed is NOT a result: run-to-run variance is ~2.5 points.

```bash
for seed in 1 2; do
  for fusion in cross self; do
    python train.py --fusion $fusion --features mfcc+gfcc --epochs 150 --seed $seed
  done
done
```
Deliverable: your `results/*.json` files sent to Person 1.

### Person 3 - the leakage experiment (needs data)
We claim published SER accuracy is inflated by evaluation protocol.
Right now that's measured with our own model. Prove it on TIM-Net itself.

TIM-Net reports 71.65% on IEMOCAP using 10-fold RANDOM CV. Our session-independent
audio baseline gets 53.86%. How much of that ~18 point gap is the split protocol
versus the architecture?

Task: run TIM-Net's own `main.py` unchanged, but with leave-one-session-out splits
instead of random 10-fold. Compare against their published 71.65%.
Deliverable: one number + a short paragraph. This is a genuine research finding.

### Person 4 - backup demo (NO dataset needed)
Standalone audio-only demo using TIM-Net's **pretrained weights** - no training.
Insurance in case the fusion model isn't ready for the presentation.

Everything needed is public:
- Repo: https://github.com/Jiaxin-Ye/TIM-Net_SER
- Pretrained weights + MFCC features: Google Drive links in their README

Environment traps (these cost us hours):
- Python 3.11, **tensorflow==2.15.1**. NOT Python 3.13, NOT TF 2.16+ (Keras 3 breaks it)
- If forced onto newer TF: `pip install tf-keras`, set `os.environ["TF_USE_LEGACY_KERAS"]="1"`
  BEFORE importing tensorflow. Verify with `tf.keras.__name__` (must contain "tf_keras").
  Do NOT check `keras.__version__` - different package, always says 3.x.
- Feature extraction must match training EXACTLY: `librosa.load()` default sr=22050
  (not 16000), pad/crop to 310000 samples centred, `n_mfcc=39`, transpose -> (606, 39)

Deliverable: working Gradio app + README.

### Person 5 - the report (NO dataset needed, start now)
Nobody should be writing this at 2am the night before.

- Methodology section: describe Systems A-E, the fusion architecture, the protocol
- Results tables from `results/*.json` as they arrive
- Integrate the 25-source literature review with what we actually did
- Write up the leakage finding (see "Findings" in README.md)

Key framing: our contribution is not "we built a pipeline". It is
**"we tested whether cross-attention beats self-attention when the audio stream
is strong, and measured how much published accuracy is evaluation artefact."**

---

## Rules that are not optional

These were measured, not guessed. Breaking them invalidates results.

1. **Never use a random train/test split.** Split by speaker or session.
   Measured leakage: +19.65 pts on RAVDESS, +7.96 pts on IEMOCAP.
2. **Do not average features over time** before the model - costs ~20 pts.
3. **Report mean +/- std over >=3 seeds.** Variance is ~2.5 pts.
4. **`TextEncoder.adapt()` on TRAINING texts only** - fitting the vocab on all
   data leaks test information.
5. Say "our audio-only baseline using the TIM-Net encoder", NOT "TIM-Net" -
   different head, different schedule, different split protocol.

---

## Getting set up

**Code:** this git repo. Data files are excluded (too large / licensed).

**Data - IMPORTANT:**
The raw IEMOCAP corpus (23 GB) is licensed from USC and **must not be copied
between laptops**. The licence forbids passing the data "or derivatives thereof"
to others, and requires everyone with access to sign it.

- Persons 2 and 3 need the data: **submit your own request** (free, few days)
  at https://sail.usc.edu/iemocap/iemocap_release.htm using your university email.
  Until approved, help Person 5 with the report.
- Person 4 needs **no licensed data** - TIM-Net's precomputed features and
  weights are published publicly by its authors. Download from their README links.

**Environment:** `./setup.sh` (needs Python 3.11).
