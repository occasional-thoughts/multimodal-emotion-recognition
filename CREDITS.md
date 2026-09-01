# Credits

This project builds directly on three pieces of published work. None of it is
redistributed here — each is fetched from its authors, and all of it should be
cited in any write-up.

## IEMOCAP — the corpus

> Busso, C., Bulut, M., Lee, C.-C., Kazemzadeh, A., Mower, E., Kim, S., Chang, J.,
> Lee, S., & Narayanan, S. (2008). *IEMOCAP: Interactive emotional dyadic motion
> capture database.* Language Resources and Evaluation, 42(4), 335–359.

Licensed from USC SAIL. Non-commercial research only; the licence requires
citation. Not distributed with this repository — see `DATA.md`.

## TIM-Net — the acoustic encoder (System B)

> Ye, J., Wen, X.-C., Wei, Y., Xu, Y., Liu, K., & Shan, H. (2023). *Temporal
> Modeling Matters: A Novel Temporal Emotional Modeling Approach for Speech
> Emotion Recognition.* ICASSP 2023.

https://github.com/Jiaxin-Ye/TIM-Net_SER — **GPL-3.0**

Used unmodified as the acoustic encoder. Our contribution is the GFCC feature
extension alongside their MFCCs, and the fusion architecture built on top.
We reproduced their published results before building anything: 92.08% on
RAVDESS (they report 90.04/90.07) and 71.65% on IEMOCAP 4-class.

## SelfCrossAttn — the research question (System D)

> Rajan, V., Brutti, A., & Cavallaro, A. (2022). *Is cross-attention preferable
> to self-attention for multi-modal emotion recognition?* ICASSP 2022.

https://github.com/smartcameras/SelfCrossAttn — no licence stated

Their finding that cross-attention and self-attention are statistically
comparable is the question this project set out to re-test with a stronger
acoustic encoder. Our fusion module implements the same comparison in Keras
rather than reusing their PyTorch code.

## Also used

* **Whisper** (OpenAI) — ASR in the live demo. MIT.
* **librosa** — MFCC extraction. ISC.
* **spafe** — GFCC extraction. BSD-3-Clause.
* **RAVDESS** (Livingstone & Russo, 2018) — used only in the evaluation-protocol
  leakage study. CC BY-NC-SA 4.0.
