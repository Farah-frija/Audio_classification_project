import torch
from torch.utils.data import Dataset
import numpy as np


# ---------------------------------------------------------------------------
# Patch-Mix helper
# ---------------------------------------------------------------------------

def patch_mix(spec_a, spec_b, patch_size=16, mix_ratio=0.3):
    """
    Patch-Mix augmentation (from Bae et al., 2023).

    Randomly replaces `mix_ratio` of the patches in spec_a with
    the corresponding patches from spec_b.

    spec_a, spec_b : (time, freq) tensors with same shape.
    patch_size     : spatial size of each patch (same as AST patch = 16).
    mix_ratio      : fraction of patches to swap (0.3 → 30% of patches).

    Why Patch-Mix instead of waveform Mixup?
      Waveform Mixup blends two raw signals linearly → can create
      unnatural artifacts when combining respiratory cycles from
      different patients. Patch-Mix operates on the spectrogram
      *after* feature extraction, swapping self-contained visual
      regions instead of blending — more aligned with how AST
      actually tokenises the input.

    Note: this version operates on the spectrogram BEFORE the processor
    returns it, so we apply it to the raw waveform space via the
    spectrogram computed by the processor. We handle this in __getitem__
    by calling the processor for both samples, then mixing the spectrograms.
    """
    T, F = spec_a.shape
    out = spec_a.clone()

    # Build list of all patch top-left corners
    patches = []
    for t in range(0, T - patch_size + 1, patch_size):
        for f in range(0, F - patch_size + 1, patch_size):
            patches.append((t, f))

    if not patches:
        return out

    n_mix = max(1, int(len(patches) * mix_ratio))
    chosen = np.random.choice(len(patches), size=n_mix, replace=False)

    for idx in chosen:
        t, f = patches[idx]
        out[t: t + patch_size, f: f + patch_size] = \
            spec_b[t: t + patch_size, f: f + patch_size]

    return out


# ---------------------------------------------------------------------------
# Dataset (Patch-Mix only)
# ---------------------------------------------------------------------------

class ASTDataset(Dataset):
    """
    ASTDataset with Patch-Mix augmentation (Step 2).
    
    Patch-Mix: swaps random spectrogram patches between two samples.
    When mixing with an abnormal sample, you synthetically inject 
    pathological signal into a normal-heavy sample — which is exactly 
    the distribution problem we need to address.
    """

    def __init__(self, X, y, device_ids, processor, train=True,
                 use_patch_mix=True, patch_mix_prob=0.5):
        self.X = X
        self.y = y
        self.device_ids = device_ids
        self.processor = processor
        self.train = train
        self.use_patch_mix = use_patch_mix
        self.patch_mix_prob = patch_mix_prob

        # Separate abnormal indices for patch-mix pairing:
        # when mixing, we prefer pairing with an abnormal sample so the
        # mixed result keeps pathological signal present
        self.abnormal_indices = np.where(np.array(y) != 0)[0]
        self.all_indices = np.arange(len(y))

    # ------------------------------------------------------------------
    # Internal: wav → spectrogram tensor
    # ------------------------------------------------------------------

    def _wav_to_spec(self, wav):
        """Run the ASTFeatureExtractor and return (time, freq) float tensor."""
        inputs = self.processor(wav, sampling_rate=16000, return_tensors="pt")
        # ASTFeatureExtractor returns shape (1, time, freq)
        return inputs.input_values.squeeze(0)  # → (time, freq)

    # ------------------------------------------------------------------
    # Internal: original waveform augmentations (kept from baseline)
    # ------------------------------------------------------------------

    def _wav_augment(self, wav):
        if np.random.random() < 0.5:
            wav = wav * np.random.uniform(0.9, 1.1)
        if np.random.random() < 0.5:
            wav = wav + np.random.normal(0, 0.0001, wav.shape)
        return wav

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        wav = self.X[idx].copy()

        if not self.train:
            # ---- Inference: no augmentation ----
            spec = self._wav_to_spec(wav)
            return spec, torch.tensor(self.y[idx], dtype=torch.long), self.device_ids[idx]

        # ---- Training path ----

        # 1. Waveform-level augmentation (original baseline)
        wav = self._wav_augment(wav)

        # 2. Compute spectrogram for this sample
        spec = self._wav_to_spec(wav)   # (time, freq)

        # 3. Patch-Mix (spectrogram-level, between two samples)
        if self.use_patch_mix and np.random.random() < self.patch_mix_prob:
            # Prefer pairing with an abnormal sample when possible
            if len(self.abnormal_indices) > 0:
                pair_idx = int(np.random.choice(self.abnormal_indices))
            else:
                pair_idx = int(np.random.choice(self.all_indices))

            wav_b = self.X[pair_idx].copy()
            wav_b = self._wav_augment(wav_b)
            spec_b = self._wav_to_spec(wav_b)

            spec = patch_mix(spec, spec_b, patch_size=16, mix_ratio=0.3)

        # 4. Return (No SpecAugment - just Patch-Mix)
        return spec, torch.tensor(self.y[idx], dtype=torch.long), self.device_ids[idx]

    def __len__(self):
        return len(self.y)