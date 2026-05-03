import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from transformers import ASTFeatureExtractor
from tqdm import tqdm
import os
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from src.dataset import ASTDataset
from src.model import CustomAST
from src.sam import SAM
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

def build_criterion(args, class_weights=None):
    """
    Choose loss function via --loss argument.

    'asl'     → AsymmetricLoss (recommended for Step 2 — maximises sensitivity)
    'focal'   → FocalLoss (lighter alternative, good baseline)
    'ce'      → CrossEntropyLoss with label smoothing (original baseline)

    class_weights : optional tensor of per-class weights, passed to FocalLoss/CE.
    """
    if args.loss == 'asl':
        print(f"   Loss: AsymmetricLoss (gamma_neg={args.asl_gamma_neg}, "
              f"gamma_pos={args.asl_gamma_pos}, clip={args.asl_clip})")
        return AsymmetricLoss(
            gamma_neg=args.asl_gamma_neg,
            gamma_pos=args.asl_gamma_pos,
            clip=args.asl_clip
        )
    elif args.loss == 'focal':
        print(f"   Loss: FocalLoss (gamma={args.focal_gamma})")
        return FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    else:
        print("   Loss: CrossEntropyLoss (label_smoothing=0.1)  [baseline]")
        return nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)


# ---------------------------------------------------------------------------
# validation function
# ---------------------------------------------------------------------------

def validate(model, val_loader, device, criterion=None):
    """
    Evaluate model on validation set.
    Returns (se, sp, score, loss)
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            
            if criterion is not None:
                loss = criterion(logits, labels)
                running_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    se, sp, score = compute_metrics(cm)
    avg_loss = running_loss / len(val_loader) if criterion is not None else 0
    
    return se, sp, score, avg_loss


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def test(args):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device: {DEVICE}")
    print(f"\n--- Testing configuration ---")
    print(f"   Loss          : {args.loss}")
    print(f"   Checkpoint    : {args.checkpoint_path}")
    print(f"----------------------------\n")

    # ------------------------------------------------------------------ data
    print(f"📥 Loading data: {args.data_path}")
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(
            f"Data file not found: {args.data_path}. Run preprocess.py first."
        )

    data = np.load(args.data_path)
    X_test, y_test, d_test = data['X_test'], data['y_test'], data['device_test']
    
    print(f"   Test samples  : {len(X_test)}")
    print(f"   Class distribution (test): {np.bincount(y_test)}")
    print()

    processor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    test_dataset = ASTDataset(
        X_test, y_test, d_test, processor, train=False
    )

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ------------------------------------------------------------------ model
    print("🧠 Building model...")
    model = CustomAST(num_classes=4).to(DEVICE)
    
    # Load trained model
    print(f"📂 Loading model from: {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    state_dict = torch.load(args.checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("✅ Model loaded successfully!")
    
    # ------------------------------------------------------------------ test
    print("\n📊 Running inference on test set...")
    
    # Evaluate on test set
    test_se, test_sp, test_score, _ = validate(model, test_loader, DEVICE)
    
    print(f"\n{'='*55}")
    print(f"🎯 FINAL TEST RESULTS:")
    print(f"   Sensitivity (Se): {test_se*100:.2f}%")
    print(f"   Specificity (Sp): {test_sp*100:.2f}%")
    print(f"   Score:            {test_score*100:.2f}%")
    print(f"{'='*55}\n")

    # Save test results
    results = {
        'test_sensitivity': test_se,
        'test_specificity': test_sp,
        'test_score': test_score,
        'checkpoint_path': args.checkpoint_path
    }
    
    if args.save_results:
        results_path = os.path.join(args.checkpoint_dir, "test_results.npy")
        np.save(results_path, np.array([results], dtype=object))
        print(f"📊 Results saved to {results_path}")
    
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test AST model on ICBHI test set"
    )

    # ---- paths ----
    parser.add_argument("--data_path",       type=str,   default="./icbhi_ast_16k_8s_metadata.npz")
    parser.add_argument("--checkpoint_dir",  type=str,   default="./checkpoints")
    parser.add_argument("--checkpoint_path", type=str,   required=True,
                        help="Path to trained model checkpoint (.pth file)")
    
    # ---- testing ----
    parser.add_argument("--batch_size",      type=int,   default=8)
    parser.add_argument("--save_results",    action="store_true",
                        help="Save test results to file")

    # ---- loss function (for compatibility) ----
    parser.add_argument("--loss",            type=str,   default="asl",
                        choices=["asl", "focal", "ce"],
                        help="Loss function used in training (for reference only)")
    
    # ASL hyperparameters (for compatibility)
    parser.add_argument("--asl_gamma_neg",   type=float, default=4.0)
    parser.add_argument("--asl_gamma_pos",   type=float, default=1.0)
    parser.add_argument("--asl_clip",        type=float, default=0.05)
    parser.add_argument("--focal_gamma",     type=float, default=2.0)
    
    # Augmentation args (for compatibility - not used in testing)
    parser.add_argument("--no_spec_augment", action="store_true")
    parser.add_argument("--no_patch_mix",    action="store_true")
    parser.add_argument("--patch_mix_prob",  type=float, default=0.5)
    parser.add_argument("--save_on_sensitivity", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    
    # Training args (for compatibility - not used)
    parser.add_argument("--epochs",          type=int,   default=20)
    parser.add_argument("--lr",              type=float, default=1e-5)
    parser.add_argument("--val_split",       type=float, default=0.2)

    args = parser.parse_args()
    
    # Run test only
    test(args)