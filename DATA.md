# Getting the data and models

This repository contains **code only**. Two categories of file are deliberately
excluded, and you will need both before anything runs.

## 1. IEMOCAP data — obtain your own licence

`data/iemocap_index.csv` and `data/iemocap_dataset.npz` are **not** in this repo,
and neither are the demo audio clips.

IEMOCAP is licensed from USC SAIL. The agreement forbids passing the data
"or derivatives thereof" to others, and requires everyone with access to read and
abide by the same terms. Transcripts and extracted features are derivatives, so
they cannot live in a shared repository.

**To get access:** submit your own request at
https://sail.usc.edu/iemocap/iemocap_release.htm using your university email.
Approval takes a few days.

Once approved, rebuild the dataset:

```bash
./env/bin/python data/iemocap_loader.py     # -> data/iemocap_index.csv
./env/bin/python data/build_dataset.py      # -> data/iemocap_dataset.npz  (~4 min)
```

Cite IEMOCAP (Busso et al., 2008) and USC SAIL in any published work — the
licence requires it.

## 2. Trained model weights — ask the team

`system_d/models/` is excluded (weights are large and are themselves derived from
licensed data). Team members share them through the project's private Drive folder.

Drop the `.json` and `.weights.h5` pairs into `system_d/models/` and the demo will
pick up whichever checkpoint scores highest.

To train your own instead:

```bash
cd system_d
../env/bin/python train.py --fusion cross --features mfcc+gfcc --epochs 150
```

Expect roughly 45 minutes per configuration on a Colab T4, and several hours on a
laptop CPU — anything involving TIM-Net is slow without a GPU.

## 3. Baseline features and weights — public

TIM-Net's own MFCC features and pretrained weights are published by its authors
and are **not** licence-restricted. Download them from the links in
`baselines/TIM-Net_SER/Code/MFCC/README.md` and `.../Test_Models/README.md`.

## Setup

```bash
./setup.sh          # needs Python 3.11 — TensorFlow 2.15 does not support 3.13
```
