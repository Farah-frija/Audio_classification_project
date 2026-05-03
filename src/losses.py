import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss — penalizes false negatives more than false positives.
    
    Key idea for ICBHI: we want to NEVER miss a sick patient (FN is deadly),
    so we down-weight the loss contribution of easy negatives (normal samples
    the model is already confident about) and keep full pressure on abnormal ones.

    gamma_neg  : focusing parameter for negative samples (easy normals get down-weighted)
    gamma_pos  : focusing parameter for positive samples (keep high gradient on abnormals)
    clip       : probability margin — predictions below `clip` for negatives are zeroed out
                 (hard thresholding of very easy negative samples)
    
    Typical values that push sensitivity:
        gamma_neg=4, gamma_pos=1, clip=0.05   (aggressive — maximizes recall)
        gamma_neg=2, gamma_pos=0, clip=0.0    (mild — closest to focal loss)
    """

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets):
        """
        logits  : (B, C) raw model output, NOT softmaxed
        targets : (B,)  integer class labels
        """
        B, C = logits.shape

        # Convert to one-hot
        targets_one_hot = torch.zeros_like(logits)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1.0)

        # Probabilities via softmax
        probs = torch.softmax(logits, dim=1)

        # ---- Positive term (abnormal classes, y=1) ----
        probs_pos = probs * targets_one_hot
        loss_pos = targets_one_hot * torch.log(probs_pos + self.eps)

        # ---- Negative term (normal class acting as negative, y=0) ----
        probs_neg = probs * (1 - targets_one_hot)

        # Hard clip: shift probabilities so very easy negatives get zero gradient
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)

        loss_neg = (1 - targets_one_hot) * torch.log(1 - probs_neg + self.eps)

        # ---- Asymmetric focusing ----
        # Positive focusing: (1 - p_pos)^gamma_pos
        if self.gamma_pos > 0:
            pt_pos = probs * targets_one_hot          # p where y=1
            loss_pos = loss_pos * ((1 - pt_pos) ** self.gamma_pos)

        # Negative focusing: (p_neg)^gamma_neg
        if self.gamma_neg > 0:
            pt_neg = probs * (1 - targets_one_hot)   # p where y=0
            loss_neg = loss_neg * (pt_neg ** self.gamma_neg)

        loss = -(loss_pos + loss_neg)
        return loss.sum(dim=1).mean()


class FocalLoss(nn.Module):
    """
    Standard Focal Loss as a lighter alternative.
    gamma=2 is the classic setting from Lin et al.
    """

    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)           # (B,)
        pt = torch.exp(-ce_loss)                     # probability of correct class
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()