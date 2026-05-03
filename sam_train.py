import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from transformers import ASTFeatureExtractor
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import confusion_matrix

from src.dataset import ASTDataset
from src.model import CustomAST
from src.esam import ESAM          # ← only change from your original
from src.losses import AsymmetricLoss, FocalLoss


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------

def compute_metrics(cm):
    """
    Returns (sensitivity, specificity, score) from a 4-class confusion matrix,
    following the official ICBHI binary protocol:
      - Abnormal = any of {Crackle, Wheeze, Both}  (rows/cols 1, 2, 3)
      - Normal   = row/col 0
    """
    se = np.sum(cm[1:, 1:]) / np.sum(cm[1:, :]) if np.sum(cm[1:, :]) > 0 else 0
    sp = cm[0, 0] / np.sum(cm[0, :]) if np.sum(cm[0, :]) > 0 else 0
    score = (se + sp) / 2
    return se, sp, score


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def build_criterion(args, class_weights=None, reduced=True):
    """
    reduced=True  → scalar loss  (used for the actual gradient update)
    reduced=False → per-sample   (used by ESAM to rank sample hardness)
    """
    if args.loss == 'asl':
        if not reduced:
            return nn.CrossEntropyLoss(reduction='none')
        print(f"   Loss: AsymmetricLoss (gamma_neg={args.asl_gamma_neg}, "
              f"gamma_pos={args.asl_gamma_pos}, clip={args.asl_clip})")
        return AsymmetricLoss(
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos,
            clip=args.asl_clip
        )
    elif args.loss == 'focal':
        if not reduced:
            return nn.CrossEntropyLoss(reduction='none')
        print(f"   Loss: FocalLoss (gamma={args.focal_gamma})")
        return FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    else:
        if not reduced:
            return nn.CrossEntropyLoss(reduction='none', label_smoothing=0.1)
        print("   Loss: CrossEntropyLoss (label_smoothing=0.1)  [baseline]")
        return nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {DEVICE}")
    print(f"\n--- Step 2 configuration ---")
    print(f"   Optimizer     : ESAM (rho={args.rho}  rho_param={args.rho_param}  beta={args.beta})")
    print(f"   Loss          : {args.loss}")
    print(f"   SpecAugment   : {not args.no_spec_augment}")
    print(f"   Patch-Mix     : {not args.no_patch_mix}")
    print(f"   Checkpoint on : {'sensitivity' if args.save_on_sensitivity else 'score'}")
    print(f"----------------------------\n")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------ data
    print(f"📥 Loading data: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Data file not found: {args.data_path}. Run preprocess.py first."
        )

    data = np.load(args.data_path)
    X_train, y_train, d_train = data['X_train'], data['y_train'], data['device_train']
    X_test,  y_test,  d_test  = data['X_test'],  data['y_test'],  data['device_test']

    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    # Weighted sampler — kept from baseline (still needed alongside ASL)
    counts  = np.bincount(y_train)
    weights = [1.0 / counts[y] for y in y_train]
    sampler = WeightedRandomSampler(weights, len(y_train))

    train_dataset = ASTDataset(
        X_train, y_train, d_train, processor, train=True
    )
    test_dataset = ASTDataset(
        X_test, y_test, d_test, processor, train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False)

    # ------------------------------------------------------------------ model
    print("🧠 Building model...")
    model = CustomAST(num_classes=4).to(DEVICE)

    # ← ESAM replaces SAM here, everything else stays the same
    optimizer = ESAM(
        model.parameters(),
        base_optimizer=torch.optim.AdamW,
        rho=args.rho,
        rho_param=args.rho_param,
        beta=args.beta,
        lr=args.lr,
        weight_decay=1e-4
    )

    # Optional: compute inverse-frequency class weights for focal/CE loss
    class_weights = None
    if args.use_class_weights and args.loss in ('focal', 'ce'):
        cw = 1.0 / (counts / counts.sum())
        cw = cw / cw.sum() * len(cw)
        class_weights = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
        print(f"   Class weights : {cw.round(2)}")

    criterion           = build_criterion(args, class_weights, reduced=True)
    criterion_unreduced = build_criterion(args, class_weights, reduced=False)

    # ------------------------------------------------------------------ train
    print("🚀 Training starts\n")

    # Track best values for two separate checkpoints:
    #   best_model.pth       → best SCORE   (same as baseline, for fair comparison)
    #   best_sensitivity.pth → best SENSITIVITY (our Step 2 target)
    best_score       = 0.0
    best_sensitivity = 0.0

    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=False
        )

        for inputs, labels, _ in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # ESAM Step 1: score every sample to find the hardest beta-fraction
            with torch.no_grad():
                loss_per_sample = criterion_unreduced(model(inputs), labels)

            hard_mask = ESAM.select_hard_samples(loss_per_sample, beta=args.beta)
            inputs_h  = inputs[hard_mask]
            labels_h  = labels[hard_mask]
            if inputs_h.shape[0] == 0:      # safety fallback
                inputs_h, labels_h = inputs, labels

            # ESAM first step — perturb rho_param fraction of weights
            loss = criterion(model(inputs_h), labels_h)
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # ESAM second step — real update at perturbed point
            criterion(model(inputs_h), labels_h).backward()
            optimizer.second_step(zero_grad=True)

            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        # ---------------------------------------------------------- evaluate
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs  = inputs.to(DEVICE)
                logits  = model(inputs)
                preds   = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
        se, sp, score = compute_metrics(cm)
        avg_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch+1:>2} | "
            f"Loss={avg_loss:.4f} | "
            f"Se={se*100:.2f}%  Sp={sp*100:.2f}%  Score={score*100:.2f}%"
        )

        history.append({'epoch': epoch+1, 'loss': avg_loss, 'se': se, 'sp': sp, 'score': score})

        # -------------------------------------------------- checkpoint: score
        if score > best_score:
            best_score = score
            path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save(model.state_dict(), path)
            print(f"    💾 Best score saved → {path}  (score={best_score*100:.2f}%)")

        # ----------------------------------------- checkpoint: sensitivity
        if se > best_sensitivity:
            best_sensitivity = se
            path = os.path.join(args.checkpoint_dir, "best_sensitivity.pth")
            torch.save(model.state_dict(), path)
            print(f"    🎯 Best sensitivity saved → {path}  (Se={best_sensitivity*100:.2f}%)")

    # ---------------------------------------------------------------- summary
    print(f"\n{'='*55}")
    print(f"  Training complete")
    print(f"  Best Score       : {best_score*100:.2f}%")
    print(f"  Best Sensitivity : {best_sensitivity*100:.2f}%")
    print(f"{'='*55}\n")

    history_path = os.path.join(args.checkpoint_dir, "training_history.npy")
    np.save(history_path, np.array(history, dtype=object))
    print(f"📊 History saved to {history_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train AST+ESAM for ICBHI — Step 2 (augmentation + ASL)"
    )

    # ---- paths ----
    parser.add_argument("--data_path",       type=str,   default="./icbhi_ast_16k_8s_metadata.npz")
    parser.add_argument("--checkpoint_dir",  type=str,   default="./checkpoints")

    # ---- training ----
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--lr",              type=float, default=1e-5)

    # ---- ESAM ----
    parser.add_argument("--rho",             type=float, default=0.05,
                        help="Perturbation radius (same meaning as SAM rho)")
    parser.add_argument("--rho_param",       type=float, default=0.5,
                        help="Fraction of parameters to perturb (0.5 = 50%%)")
    parser.add_argument("--beta",            type=float, default=0.5,
                        help="Fraction of hardest samples used per step (0.5 = top 50%%)")

    # ---- loss function ----
    parser.add_argument("--loss",            type=str,   default="asl",
                        choices=["asl", "focal", "ce"])
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--asl_gamma_neg",   type=float, default=4.0)
    parser.add_argument("--asl_gamma_pos",   type=float, default=1.0)
    parser.add_argument("--asl_clip",        type=float, default=0.05)
    parser.add_argument("--focal_gamma",     type=float, default=2.0)

    # ---- augmentation ----
    parser.add_argument("--no_spec_augment", action="store_true")
    parser.add_argument("--no_patch_mix",    action="store_true")
    parser.add_argument("--patch_mix_prob",  type=float, default=0.5)

    # ---- checkpointing ----
    parser.add_argument("--save_on_sensitivity", action="store_true")

    args = parser.parse_args()
    train(args)