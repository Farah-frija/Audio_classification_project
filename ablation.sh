#!/bin/bash
# ablation.sh
# Runs four targeted experiments to isolate the contribution of each Step 2 change.
# Each experiment saves to its own checkpoint sub-folder.
#
# Usage:  bash ablation.sh
#
# Results to compare:
#   exp1_baseline       → ce loss, no SpecAugment, no Patch-Mix  (reproduces paper)
#   exp2_asl_only       → ASL only, no augmentation
#   exp3_aug_only       → SpecAugment + Patch-Mix, CE loss
#   exp4_full_step2     → ASL + SpecAugment + Patch-Mix          (our proposal)

DATA="./icbhi_ast_16k_8s_metadata.npz"
EPOCHS=15
BS=8
LR=1e-5
echo "========================================"
echo " EXP 1 — Partial Step  (ASL)"
echo "========================================"
python train.py --loss asl --asl_gamma_neg 4 --epochs 4 --checkpoint_dir ./checkpoints/asl_more_epocs --no_patch_mix --patch_mix_prob

echo "========================================"
echo " EXP 2 — Partial Step 2 (ASL + esam + PatchMix)"
echo "========================================"
python train.py --loss asl --asl_gamma_neg 4 --epochs 4 --checkpoint_dir ./checkpoints/PatchMix_more_epocs 


echo "========================================"
echo " EXP 3 — Full Step  (ASL + esam + PatchMix)"
echo "========================================"
python esam_train.py --loss asl --asl_gamma_neg 4 --epochs 4 --checkpoint_dir ./checkpoints/esam_more_epocs



echo ""
echo "========================================"
echo " All experiments done."
echo " Compare best_sensitivity.pth results across exp folders."
echo "========================================"