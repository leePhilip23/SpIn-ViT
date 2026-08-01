import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ViTSAELoss(nn.Module):
    def __init__(self, num_cls, temperature=0.07):
        """
        Initializes the loss function with specified weighting factors for different components.
        Args:
            lambda1 (float, optional): Weighting factor for the first loss component. Defaults to 1.
            lambda2 (float, optional): Weighting factor for the second loss component. Defaults to 1.
            lambda3 (float, optional): Weighting factor for the third loss component. Defaults to 2.
        Attributes:
            lambda1 (float): Stores the weighting factor for the first loss component.
            lambda2 (float): Stores the weighting factor for the second loss component.
            lambda3 (float): Stores the weighting factor for the third loss component.
            supconloss (SupConLoss): Instance of the SupConLoss class used for supervised contrastive loss.
        """
        super(ViTSAELoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()



    def center_loss(self, features, labels, temperature=0.07):
        """
        features: (B, D) CLS embeddings
        labels:   (B,)
        """
        device = features.device
        B, D = features.shape
        num_classes = labels.max().item() + 1

        # normalize
        features = F.normalize(features, dim=1)

        # compute class prototypes
        prototypes = torch.zeros(num_classes, D, device=device)
        counts = torch.zeros(num_classes, device=device)

        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                prototypes[c] = features[mask].mean(dim=0)
                counts[c] = mask.sum()

        prototypes = F.normalize(prototypes, dim=1)

        # similarity (B, C)
        logits = features @ prototypes.T / temperature

        loss = F.cross_entropy(logits, labels)
        return loss



    def intra_ortho(self, z):
        """
        z: (B, C, W, P)
        B = batch
        C = class windows
        W = neurons per class window
        P = patch tokens
        Enforces orthogonality among the W rows inside each class window.
        """
        B, C, W, P = z.shape
        z = F.normalize(z, p=2, dim=-1)  # (B, C, W, P)
        z_flat = z.view(B * C, W, P)     # (B*C, W, P)
        gram = torch.bmm(z_flat, z_flat.transpose(1, 2))  # (B*C, W, W)

        I = torch.eye(W, device=z.device).unsqueeze(0)    # (1, W, W)

        loss = ((gram - I) ** 2).mean()
        return loss

    

    def forward(self, vit_features, sparse_logits, reconstructed, cls_logits, labels, num_classes):
        """
        Computes the total loss as a weighted sum of entropy loss, sparsity loss, 
        and supervised contrastive loss.
        Args:
            sparse_rep (Tensor): The sparse representation of the input data.
            labels (Tensor): Ground truth labels for supervised contrastive loss.
        Returns:
            Tensor: The total computed loss.
        """
        ce_loss =  self.ce(cls_logits, labels)
        mse_loss = self.mse(reconstructed, vit_features)
        sparse_loss = torch.mean(torch.abs(sparse_logits))
        B, P, _ = sparse_logits.shape
        sparse_logits = sparse_logits.transpose(1, 2).contiguous().view(B, num_classes, -1, P)
        return ce_loss + (mse_loss + 5e-4 * sparse_loss) + 0.3 * self.center_loss(cls_logits, labels) + 2.0 * self.intra_ortho(sparse_logits)