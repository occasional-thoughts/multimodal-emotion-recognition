# Multimodal Emotion Recognition — Audio + Text Fusion

Dual-stream (audio + text) emotion recognition on IEMOCAP, 4-class
(angry / happy / neutral / sad, 5,531 utterances).

## Setup

```bash
./scripts/fetch_baselines.sh   # clones TIM-Net and SelfCrossAttn from their authors
./setup.sh                     # needs Python 3.11 — TF 2.15 does not support 3.13
```

Then see **DATA.md** — IEMOCAP is licensed and not included, so you will need
your own approved copy before anything runs.

Third-party code and the corpus are not redistributed here. See **CREDITS.md**
for what this builds on and how to cite it.

## Research question

Rajan et al. (ICASSP 2022) found cross-attention does **not** beat self-attention
for audio+text fusion — self-attention was significantly better (WA .518 vs .501,
7-class). Their audio branch was weak (audio alone = .365, the weakest modality).

**Hypothesis:** cross-attention underperforms because the audio stream is weak.
With a strong audio encoder (TIM-Net) and a richer acoustic feature set
(MFCC + GFCC), does that conclusion change?

## Architecture

| System | Role | Implementation | Output |
|---|---|---|---|
| A | data prep | `data/iemocap_loader.py`, `data/build_dataset.py` | aligned index + features |
| B | acoustic | TIM-Net (Ye et al., ICASSP 2023), unmodified Keras | seq (8, 39) + pooled (39) |
| C | semantic | `system_c/text_encoder.py` (Embedding + BiGRU) | seq (32, 128) + pooled + mask |
| D | fusion | cross- vs self-attention (Rajan's idea, in Keras) | 4-class softmax |

## Baseline (reproduced, not just cited)

| Model | Dataset | Published | Reproduced here |
|---|---|---|---|
| TIM-Net | IEMOCAP 4-class | ~68–72% | **71.65%** |
| TIM-Net | RAVDESS 8-class | 90.04 / 90.07 | **92.08%** |

```bash
cd baselines/TIM-Net_SER/Code
../../../env/bin/python main.py --mode test --data IEMOCAP \
    --test_path ./Test_Models/IEMOCAP_16 --split_fold 10 --random_seed 16
```

Note: `--mode test` loads pretrained weights and runs inference only. It verifies
their published numbers; it is not a from-scratch retrain. Any architecture change
(e.g. adding GFCC) invalidates those weights and requires real training.

## Data

`data/iemocap_dataset.npz` (698 MB) — everything aligned by utterance:

| key | shape | notes |
|---|---|---|
| `mfcc` | (5531, 606, 39) | **byte-identical to TIM-Net's own features** (verified 12/12) |
| `gfcc` | (5531, 606, 13) | our addition (Gap 2) |
| `utt_id`, `emotion`, `speaker`, `session`, `transcript` | (5531,) | |

TIM-Net's shipped `IEMOCAP.npy` has no utterance IDs, so transcripts cannot be
joined to it. We regenerate the features with their exact settings
(librosa sr=22050, pad/crop to 310000, n_mfcc=39) and verified ours match theirs.

## Experimental rules (measured, not assumed)

1. **Never use a random train/test split.** Split by speaker or session.
   Measured leakage: **+19.65 pts** on RAVDESS, **+7.96 pts** on IEMOCAP.
2. **Do not average features over time** before the model — costs ~20 pts.
3. **Report mean ± std over ≥5 seeds** — run-to-run variance is ~2.5 pts.
4. **`TextEncoder.adapt()` on training texts only** — fitting the vocabulary on
   all data leaks test information.

## Findings so far

- Leakage decomposition (`experiments/`): on RAVDESS the inflation comes from
  near-duplicate retakes (every condition recorded twice), **not** speaker overlap.
  Twin clips are 5x more similar to each other (0.600 cosine) than to another
  speaker's same emotion (0.128).
- GFCC contributes **+1.57** (IEMOCAP) / **+1.67** (RAVDESS) — small but replicated.
- IEMOCAP's standard leave-one-session-out protocol is honest; RAVDESS 10-fold is not.

## Layout

```
data/         index, aligned dataset, extraction scripts
system_b/     (TIM-Net used directly from baselines/; PyTorch port kept but UNUSED)
system_c/     text encoder
system_d/     fusion — cross vs self attention  [next]
baselines/    TIM-Net_SER (weights + features), SelfCrossAttn (Rajan reference)
experiments/  leakage studies, attention demo
```
