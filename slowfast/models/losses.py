#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Loss functions."""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pytorchvideo.losses.soft_target_cross_entropy import SoftTargetCrossEntropyLoss
except ImportError:
    class SoftTargetCrossEntropyLoss(nn.Module):
        def __init__(self, normalize_targets=False, reduction="mean"):
            super().__init__()
            self.reduction = reduction

        def forward(self, x, target, context=None):
            log_probs = F.log_softmax(x, dim=1)
            loss = -(target * log_probs).sum(dim=1)
            if self.reduction == "mean":
                return loss.mean()
            if self.reduction == "sum":
                return loss.sum()
            return loss


class CrossEntropyLossWrapper(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, labels, context=None):
        return F.cross_entropy(inputs, labels, reduction=self.reduction)


class BCELossWrapper(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.loss = nn.BCELoss(reduction=reduction)

    def forward(self, inputs, labels, context=None):
        return self.loss(inputs, labels)


class BCEWithLogitsLossWrapper(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, inputs, labels, context=None):
        return self.loss(inputs, labels)


class SoftCrossEntropyLossWrapper(nn.Module):
    def __init__(self, normalize_targets=False, reduction="mean"):
        super().__init__()
        self.normalize_targets = normalize_targets
        self.reduction = reduction

    def forward(self, inputs, targets, context=None):
        if self.normalize_targets:
            targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_probs = F.log_softmax(inputs, dim=1)
        loss = -(targets * log_probs).sum(dim=1)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(ContrastiveLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, dummy_labels=None, context=None):
        targets = torch.zeros(inputs.shape[0], dtype=torch.long, device=inputs.device)
        return F.cross_entropy(inputs, targets, reduction=self.reduction)


class MultipleMSELoss(nn.Module):
    """
    Compute multiple mse losses and return their average.
    """

    def __init__(self, reduction="mean"):
        super(MultipleMSELoss, self).__init__()
        self.mse_func = nn.MSELoss(reduction=reduction)

    def forward(self, x, y, context=None):
        loss_sum = 0.0
        multi_loss = []
        for xt, yt in zip(x, y):
            if isinstance(yt, (tuple,)):
                if len(yt) == 2:
                    yt, wt = yt
                    lt = "mse"
                elif len(yt) == 3:
                    yt, wt, lt = yt
                else:
                    raise NotImplementedError
            else:
                wt, lt = 1.0, "mse"
            if lt == "mse":
                loss = self.mse_func(xt, yt)
            else:
                raise NotImplementedError
            loss_sum += loss * wt
            multi_loss.append(loss)
        return loss_sum, multi_loss


class CEMFormerContextConsistencyCrossEntropyLoss(nn.Module):
    """
    Paper-faithful adaptation for DAAD / M2MVT.

    Label order assumed:
        0 ST
        1 RT
        2 LT
        3 RLC
        4 LLC
        5 SS
        6 UT

    Context vector:
        [leftmost_lane, rightmost_lane, near_intersection]

    CCL is only applied to the maneuver classes that CEMFormer defines.
    SS and UT are left as CE-only because the papers do not specify
    contradiction rules for them.
    """

    def __init__(self, reduction="mean", cc_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.cc_weight = cc_weight
        self.eps = 1e-7

    @staticmethod
    def _to_context_tensor(context, device):
        if context is None:
            return None

        if isinstance(context, dict):
            for key in ("traffic_context", "context", "ctx", "meta_context"):
                if key in context:
                    context = context[key]
                    break
            else:
                return None

        if torch.is_tensor(context):
            ctx = context
        else:
            ctx = torch.as_tensor(context)

        ctx = ctx.to(device=device)
        if ctx.ndim == 1:
            ctx = ctx.unsqueeze(0)
        return ctx.float()

    def _cc_loss(self, logits, labels, context):
        probs = F.softmax(logits, dim=1).clamp(self.eps, 1.0 - self.eps)
        loss = logits.new_zeros(())

        rules = {
            1: [  # RT
                lambda c: (c[:, 0] == 1) & (c[:, 1] == 0),
                lambda c: (c[:, 2] == 0),
            ],
            2: [  # LT
                lambda c: (c[:, 0] == 0) & (c[:, 1] == 1),
                lambda c: (c[:, 2] == 0),
            ],
            3: [  # RLC
                lambda c: (c[:, 0] == 1),
            ],
            4: [  # LLC
                lambda c: (c[:, 1] == 1),
            ],
        }

        for cls_idx, predicates in rules.items():
            cls_mask = labels == cls_idx
            if not torch.any(cls_mask):
                continue

            p_cls = probs[cls_mask, cls_idx]
            ctx = context[cls_mask]

            for pred in predicates:
                violated = pred(ctx)
                if torch.any(violated):
                    loss = loss - torch.log1p(-p_cls[violated]).sum()

        if self.reduction == "mean":
            loss = loss / max(labels.shape[0], 1)
        return loss

    def forward(self, inputs, labels, context=None):
        ce = F.cross_entropy(inputs, labels, reduction=self.reduction)
        ctx = self._to_context_tensor(context, inputs.device)
        if ctx is None:
            return ce
        cc = self._cc_loss(inputs, labels, ctx)
        return ce + self.cc_weight * cc


_LOSSES = {
    "cross_entropy": CrossEntropyLossWrapper,
    "bce": BCELossWrapper,
    "bce_logit": BCEWithLogitsLossWrapper,
    "soft_cross_entropy": partial(SoftCrossEntropyLossWrapper, normalize_targets=False),
    "contrastive_loss": ContrastiveLoss,
    "mse": nn.MSELoss,
    "multi_mse": MultipleMSELoss,
    "cemformer_ce_cc": CEMFormerContextConsistencyCrossEntropyLoss,
    "m2mvt_ce_cc": CEMFormerContextConsistencyCrossEntropyLoss,
}


def get_loss_func(loss_name):
    """
    Retrieve the loss given the loss name.
    Args (int):
        loss_name: the name of the loss to use.
    """
    if loss_name not in _LOSSES.keys():
        raise NotImplementedError("Loss {} is not supported".format(loss_name))
    return _LOSSES[loss_name]