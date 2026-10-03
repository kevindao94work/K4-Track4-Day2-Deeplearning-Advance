"""Loss nhiều lớp và trộn cả ảnh lẫn nhãn; trọng số chỉ lấy từ train."""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_criterion(kind: str = 'ce', **kw):
    if kind == 'ce':
        return nn.CrossEntropyLoss()
    if kind == 'ls':
        return LabelSmoothingCE(kw.get('smoothing', .1))
    if kind == 'focal':
        return FocalLoss(kw.get('gamma', 2.), kw.get('alpha'))
    if kind == 'ce_weighted':
        if kw.get('weight') is None:
            raise ValueError('CE có trọng số cần weight tính từ train')
        return nn.CrossEntropyLoss(weight=kw['weight'])
    raise ValueError(f'Loss không hợp lệ: {kind}')


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = .1):
        super().__init__()
        if not 0 <= smoothing < 1:
            raise ValueError('Smoothing phải nằm trong [0,1)')
        self.smoothing = smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2., alpha=None):
        super().__init__()
        if gamma < 0 or not math.isfinite(gamma):
            raise ValueError('Gamma phải hữu hạn và không âm')
        self.gamma = gamma
        if alpha is not None:
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
            if alpha.ndim != 1 or not torch.isfinite(alpha).all() or (alpha < 0).any() or alpha.sum() <= 0:
                raise ValueError('Alpha phải là vector trọng số không âm')
        self.register_buffer('alpha', alpha)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        logpt = logp.gather(1, target[:, None]).squeeze(1)
        factor = (1 - logpt.exp()).pow(self.gamma)
        loss = -logpt * factor
        if self.alpha is None:
            return loss.mean()
        weight = self.alpha[target]
        # Chuẩn hóa như weighted CE; gamma=0 tương đương cả weighted CE.
        return (weight * loss).sum() / weight.sum().clamp_min(1e-12)


def class_weights(counts, beta: float = 0.):
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.shape != (9,) or not torch.isfinite(counts).all() or (counts <= 0).any():
        raise ValueError('Cần 9 số đếm train dương và hữu hạn')
    if not 0 <= beta < 1:
        raise ValueError('Beta phải nằm trong [0,1)')
    if beta == 0:
        weights = counts.reciprocal()
    else:
        weights = (1 - beta) / (-torch.expm1(counts * math.log(beta)))
    return (weights / weights.mean()).float()


def weights_from_train(df, beta=0.):
    if df.attrs.get('split') != 'train' or df.attrs.get('fold') != 0:
        raise ValueError('Chỉ được tính trọng số lớp từ train fold 0')
    return class_weights(df.Label.value_counts().reindex(range(9), fill_value=0).to_numpy(), beta)


def mix_batch(x, y, alpha: float = 1., mode: str = 'cutmix'):
    if mode not in ('mixup', 'cutmix') or alpha < 0:
        raise ValueError('Mixup/CutMix cần mode hợp lệ và alpha không âm')
    if x.ndim != 4 or y.ndim != 1 or len(x) != len(y) or len(y) == 0 or x.device != y.device:
        raise ValueError('Cần batch NCHW và nhãn N trên cùng thiết bị')
    if alpha == 0:
        return x, (y, y, 1.)
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(len(y), device=y.device)
    if mode == 'mixup':
        mixed = lam * x + (1 - lam) * x[perm]
    else:
        h, w = x.shape[-2:]
        ratio = math.sqrt(1 - lam)
        ch, cw = int(h * ratio), int(w * ratio)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(0, cy - ch // 2), min(h, cy + ch // 2)
        x1, x2 = max(0, cx - cw // 2), min(w, cx + cw // 2)
        mixed = x.clone()
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1)) / (h * w)
    return mixed, (y, y[perm], lam)


def soft_targets(targets, num_classes=9):
    ya, yb, lam = targets
    return lam * F.one_hot(ya, num_classes).float() + (1 - lam) * F.one_hot(yb, num_classes).float()


def mixed_loss(criterion, logits, targets):
    ya, yb, lam = targets
    if not 0 <= lam <= 1:
        raise ValueError('Lambda ngoài [0,1]')
    return lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb)
