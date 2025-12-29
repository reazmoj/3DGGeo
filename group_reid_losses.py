import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupReIDLoss(nn.Module):
    """
    Joint loss for Group Re-Identification:
    - Person-level: Cross-Entropy + Batch-Hard Triplet
    - Group-level:  Cross-Entropy + Batch-Hard Triplet
    """

    def __init__(self, alpha=1.0, beta=1.0, margin=0.7):
        super(GroupReIDLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.margin = margin
        self.alpha = alpha  # weight for triplet loss
        self.beta = beta   # weight for person-level loss

    def forward(
        self,
        person_feats, person_logits,
        group_feats, group_logits,
        person_labels, group_labels,
        person_mask=None,
        return_components=False
    ):
        """
        Args:
            person_feats:  (B*N, D)
            person_logits: (B*N, C_p)
            group_feats:   (B, D)
            group_logits:  (B, C_g)
            person_labels: (B*N,)
            group_labels:  (B,)
            person_mask:   (B*N,)  True for valid persons
        """

        # -------------------------
        # Normalize features
        # -------------------------
        person_feats = F.normalize(person_feats, p=2, dim=1)
        group_feats = F.normalize(group_feats, p=2, dim=1)

        # -------------------------
        # Person-level loss
        # -------------------------
        if person_mask is None:
            # fallback (not recommended, but safe)
            person_mask = person_labels >= 0

        valid_person_feats = person_feats[person_mask]
        valid_person_logits = person_logits[person_mask]
        valid_person_labels = person_labels[person_mask]

        if valid_person_labels.numel() > 1:
            L_cross_person = self.cross_entropy(
                valid_person_logits,
                valid_person_labels
            )
            L_triplet_person = self._batch_hard_triplet_loss(
                valid_person_feats,
                valid_person_labels
            )
        else:
            device = person_feats.device
            L_cross_person = torch.tensor(0.0, device=device)
            L_triplet_person = torch.tensor(0.0, device=device)

        L_p = L_cross_person + self.alpha * L_triplet_person

        # -------------------------
        # Group-level loss
        # -------------------------
        L_cross_group = self.cross_entropy(group_logits, group_labels)
        L_triplet_group = self._batch_hard_triplet_loss(
            group_feats,
            group_labels
        )

        L_g = L_cross_group + self.alpha * L_triplet_group

        # -------------------------
        # Total loss
        # -------------------------
        L_all = L_g + self.beta * L_p

        if return_components:
            return (
                L_all,
                L_cross_person,
                L_triplet_person,
                L_cross_group,
                L_triplet_group
            )

        return L_all

    def _batch_hard_triplet_loss(self, features, labels):
        """
        Batch-Hard Triplet Loss
        """
        if labels.numel() < 2:
            return torch.tensor(0.0, device=features.device)

        # Pairwise distance
        dist_mat = torch.cdist(features, features, p=2)

        labels = labels.view(-1, 1)

        pos_mask = labels.eq(labels.t())
        neg_mask = ~pos_mask

        pos_mask.fill_diagonal_(False)

        hardest_pos = (dist_mat * pos_mask.float()).max(dim=1)[0]
        hardest_neg = (dist_mat + (~neg_mask).float() * 1e6).min(dim=1)[0]

        loss = F.relu(hardest_pos - hardest_neg + self.margin)
        return loss.mean()