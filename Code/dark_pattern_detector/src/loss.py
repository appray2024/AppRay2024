import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations


class ContrastiveLoss(nn.Module):
    def __init__(self, n_negative=16, weight=None):
        super(ContrastiveLoss, self).__init__()
        self.n_negative = n_negative
        self.class_weight = weight

    def forward(self, xs, y_preds, y_true, use_contrastive_loss=False):
        loss_ce = F.cross_entropy(y_preds, y_true, weight=self.class_weight)

        if not use_contrastive_loss:
            return loss_ce

        loss_pos = self.compute_positive_loss(xs, y_true)
        loss_neg = self.compute_negative_loss(xs, y_true, len(loss_pos))

        total_loss = loss_ce + loss_pos + loss_neg
        return total_loss

    def compute_positive_loss(self, xs, y_true):
        loss_pos = []
        pos_pairs = []

        _, inv_indices, counts = torch.unique(
            y_true, return_inverse=True, return_counts=True
        )
        duplicate_indices = [
            torch.where(inv_indices == i)[0].tolist()
            for i, count in enumerate(counts)
            if count > 1
        ]

        for dups in duplicate_indices:
            pair_combinations = list(combinations(dups, 2))
            pos_pairs.extend(pair_combinations)

            loss_pos.extend(
                1 - F.cosine_similarity(xs[c[0]], xs[c[1]], dim=0)
                for c in pair_combinations
            )

        if loss_pos:
            loss_pos = torch.mean(torch.stack(loss_pos))
        else:
            loss_pos = torch.tensor(0.0, device=xs.device)

        return loss_pos

    def compute_negative_loss(self, xs, y_true, n_pos_pairs):
        if len(y_true) < 2:
            return torch.tensor(0.0, device=xs.device)

        all_pairs = list(combinations(range(len(y_true)), 2))
        negative_pairs = [
            pair for pair in all_pairs if pair not in self.compute_positive_loss
        ]

        n_negative = min(self.n_negative, len(negative_pairs))
        negative_pairs = negative_pairs[:n_negative]

        loss_neg = torch.stack(
            [
                torch.max(
                    torch.tensor(0.0, device=xs.device),
                    F.cosine_similarity(xs[pair[0]], xs[pair[1]], dim=0),
                )
                for pair in negative_pairs
            ]
        )

        return (
            torch.mean(loss_neg)
            if loss_neg.size(0) > 0
            else torch.tensor(0.0, device=xs.device)
        )
