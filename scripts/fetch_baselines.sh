#!/bin/bash
# Clones the two research repositories this project builds on.
# They are NOT redistributed here — TIM-Net is GPL-3.0 and SelfCrossAttn
# carries no licence, so each is fetched from its authors directly.
set -e
cd "$(dirname "$0")/../baselines"

[ -d TIM-Net_SER ]   || git clone --depth 1 https://github.com/Jiaxin-Ye/TIM-Net_SER.git
[ -d SelfCrossAttn ] || git clone --depth 1 https://github.com/smartcameras/SelfCrossAttn.git

cat <<'NOTE'

Cloned. Two more downloads are needed for the TIM-Net baseline, from the
Google Drive links in its own README:

  TIM-Net_SER/Code/MFCC/         precomputed features
  TIM-Net_SER/Code/Test_Models/  pretrained weights (unzip them)

Both are published by TIM-Net's authors and are not licence-restricted.
NOTE
