#from info_nce import InfoNCE, info_nce
from infonce import SupervisedInfoNCE
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSupConLoss(nn.Module):
    """
    Supervised Contrastive Loss adapted for soft labels.
    Accepts soft label similarities instead of hard class labels.
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SoftSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, soft_labels=None):
        """
        Args:
            features: Tensor of shape [bsz, n_views, ...]
            soft_labels: Tensor of shape [bsz, num_classes] with class probabilities
        Returns:
            Scalar loss
        """
        device = features.device
        if len(features.shape) < 3:
            raise ValueError("`features` must be [bsz, n_views, ...]")

        bsz, n_views = features.shape[:2]
        features = features.view(bsz, n_views, -1)

        # Build soft similarity matrix
        if soft_labels is not None:
            soft_labels = F.normalize(soft_labels, p=1, dim=1)  # Ensure probs sum to 1
            mask = torch.matmul(soft_labels, soft_labels.T)  # shape: [bsz, bsz]
        else:
            mask = torch.eye(bsz, dtype=torch.float32, device=device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz * n_views, feat_dim]
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = n_views
        else:
            raise ValueError(f"Unknown contrast_mode: {self.contrast_mode}")

        # Compute logits
        anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        logits_max, _ = anchor_dot_contrast.max(dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Expand mask for all anchors
        mask = mask.repeat(anchor_count, n_views)
        logits_mask = torch.ones_like(mask)
        diag_idx = torch.arange(bsz * anchor_count, device=device)
        logits_mask[diag_idx, diag_idx] = 0
        mask *= logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum < 1e-6, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, bsz).mean()

        return loss
    

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss
    

class OrthogonalBatchLoss(nn.Module):
    def __init__(self, num_classes: int):
        super(OrthogonalBatchLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, features):
        batch_size, feature_dim = features.shape
        segment_size = feature_dim // self.num_classes
        assert feature_dim % self.num_classes == 0, "Feature dim must be divisible by number of classes."
        segments = torch.chunk(features, self.num_classes, dim=1)
        stacked = torch.stack(segments, dim=0)
        stacked_norm = F.normalize(stacked, dim=2)
        class_vectors = stacked_norm.mean(dim=1)
        cosine_sim = torch.matmul(class_vectors, class_vectors.T)
        mask = torch.eye(self.num_classes, device=features.device).bool()
        cosine_sim = cosine_sim.masked_fill(mask, 0.0)
        orth_loss = cosine_sim.abs().sum() / (self.num_classes * (self.num_classes - 1))
        return orth_loss
    

class OrthogonalBatchLoss2(nn.Module):
    def __init__(self):
        super(OrthogonalBatchLoss2, self).__init__()

    def forward(self, activations, segments):
        """
        activations: Tensor of shape (batch_size, feature_dim)
        segments: List[Tensor] of length num_classes, each tensor is (n_indices_per_class,)
        
        Goal: Make class-specific segment activations orthogonal across classes
        """
        num_classes = len(segments)
        projections = []

        for i in range(num_classes):
            indices = segments[i]
            proj = activations[:, indices]  # Shape: (batch_size, n_indices_per_class)
            projections.append(F.normalize(proj, dim=1))  # Normalize per sample

        # Stack and compute pairwise dot products
        projections = torch.stack(projections)  # Shape: (num_classes, batch_size, n_indices)
        ortho_loss = 0.0
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                dot = torch.einsum('bd,bd->b', projections[i], projections[j])  # Dot per sample
                ortho_loss += (dot ** 2).mean()  # Penalize squared cosine similarity

        return ortho_loss
    

class OrthogonalLossFunction(nn.Module):
    def __init__(self):
        super(OrthogonalLossFunction, self).__init__()

    def forward(self, latent: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
        """
        Computes orthogonal loss over a batch of latent vectors based on segment indices.

        Args:
            latent (torch.Tensor): Latent representations, shape (batch_size, latent_dim)
            segments (torch.Tensor): Indices into the latent vector, shape (batch_size, k)

        Returns:
            torch.Tensor: Scalar loss encouraging orthogonality among selected neurons.
        """
        batch_size, latent_dim = latent.shape
        device = latent.device
        k = segments.shape[1]

        loss_batch = []

        for i in range(batch_size):
            indices = segments[i]  # Shape: (k,)
            selected = latent[i, indices]  # Shape: (k,)
            # Normalize to unit vectors
            normed = F.normalize(selected, dim=0)  # Shape: (k,)
            # Compute cosine similarity matrix
            cos_sim_matrix = torch.matmul(normed.unsqueeze(1), normed.unsqueeze(0))  # (k, k)
            identity = torch.eye(k, device=device)
            off_diag_cos = cos_sim_matrix * (1 - identity)  # remove diagonal (self-similarity)
            num_off_diag = k * (k - 1)
            loss = off_diag_cos.sum() / num_off_diag
            loss_batch.append(loss)

        return torch.stack(loss_batch).mean()
    


class VanillaSAELoss(nn.Module):
    def __init__(self, lambda_l1=8e-5):
        """
        Initialize the Sparse Autoencoder Loss function.
        
        Args:
            lambda_l1 (float): The L1 regularization coefficient for sparsity.
        """
        super(VanillaSAELoss, self).__init__()
        self.lambda_l1 = lambda_l1
        #self.supconloss = SupConLoss()
        
    def forward(self, reconstructed, original, activations):
        """
        Calculate the SAE loss: L_SAE = ||SAE(z) - z||^2 + λ_l1 ||φ(f(z))||_1
        
        Args:
            reconstructed (Tensor): The output of the autoencoder, SAE(z)
            original (Tensor): The original input, z
            activations (Tensor): The hidden layer activations, φ(f(z))
        
        Returns:
            Tensor: The computed loss value
        """
        reconstruction_loss = F.mse_loss(reconstructed, original)
        sparsity_loss = self.lambda_l1 * torch.linalg.vector_norm(activations, ord=1, dim=1).mean()
        return reconstruction_loss + sparsity_loss


class ViTSAELoss(nn.Module):
    def entropy_loss(self, sparse_rep):
        """
        Computes the entropy loss for a given sparse representation.
        The loss is calculated as the mean of the binary entropy for each element
        in the sparse representation tensor.
        Args:
            sparse_rep (torch.Tensor): The sparse representation tensor.
        Returns:
            torch.Tensor: The computed entropy loss.
        """
        return -(
            sparse_rep * torch.log(sparse_rep + 1e-8) +
            (1 - sparse_rep) * torch.log(1 - sparse_rep + 1e-8)
        ).mean()


    def sparse_loss(self, sparse_rep):
        """
        Computes the sparse loss for a given sparse representation.
        The sparse loss is calculated as the mean of the L1 norm 
        (sum of absolute values) of the sparse representation vectors.
        Args:
            sparse_rep (torch.Tensor): A tensor representing the sparse 
                representation, with shape (batch_size, ...).
        Returns:
            torch.Tensor: A scalar tensor representing the mean sparse loss.
        """

        return torch.mean(torch.abs(sparse_rep)) #torch.linalg.vector_norm(sparse_rep, ord=1, dim=1).mean() #torch.mean(torch.abs(sparse_rep)) 
    

    def sup_con_loss(self, features, labels):
        """
        Compute the supervised contrastive loss.
        Args:
            features (torch.Tensor): Normalized feature embeddings of shape (batch_size, feature_dim).
            labels (torch.Tensor): Ground truth labels of shape (batch_size,).
        Returns:
            torch.Tensor: Scalar loss value.
        Notes:
            - The loss encourages samples with the same label to have similar embeddings.
            - Uses temperature scaling for logits normalization.
        """
        features = F.normalize(features, dim=1)
        device = features.device
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
 
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
 
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
        mask = mask * logits_mask
 
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
 
        return -mean_log_prob_pos.mean()
    

    def sup_con_loss2(self, features, labels):
        """
        Compute the supervised contrastive loss.
        Args:
            features (torch.Tensor): Normalized feature embeddings of shape (batch_size, feature_dim).
            labels (torch.Tensor): Ground truth labels of shape (batch_size,).
        Returns:
            torch.Tensor: Scalar loss value.
        Notes:
            - The loss encourages samples with the same label to have similar embeddings.
            - Uses temperature scaling for logits normalization.
        """
        batch_size, n_views, features_dim = features.shape
        features = F.normalize(features, dim=1).view(batch_size, -1)


        device = features.device
        labels = labels.contiguous().view(-1, 1)
        #labels = labels.view(-1, 1)
        #labels = labels.repeat(1, n_views).view(batch_size, -1)
        mask = torch.eq(labels, labels.T).float().to(device)
 
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
 
        logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(device)
        mask = mask * logits_mask
 
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
 
        return -mean_log_prob_pos.mean()
    

    def weighted_bce_loss(self, preds, targets, pos_weight=1.0, neg_weight=1.0):
        """
        preds: predicted logits or probabilities (after sigmoid)
        targets: binary ground truth (0 or 1)
        pos_weight: weight for positive (target == 1)
        neg_weight: weight for negative (target == 0)
        """
        # Ensure predictions are in the (0, 1) range
        preds = torch.clamp(preds, min=1e-7, max=1 - 1e-7)
        # Compute weighted BCE manually
        loss = - (pos_weight * targets * torch.log(preds) +
                neg_weight * (1 - targets) * torch.log(1 - preds))
        return loss.mean()
    

    def compute_hinton_loss(self, outputs, t_outputs,  kd_temp=3):
        soft_label = F.softmax(t_outputs / kd_temp, dim=1)
        kd_loss = nn.KLDivLoss(reduction='batchmean')(F.log_softmax(outputs / kd_temp, dim=1),
                                                    soft_label) * (kd_temp * kd_temp)
        return kd_loss
    

    def compute_kl_loss(self, logits, target_logits):
        # normalize both
        log_q = F.log_softmax(logits, dim=1)      # log probabilities for KL input
        p = F.softmax(target_logits, dim=1)       # probability distribution target

        # compute KL divergence loss
        kl_loss = F.kl_div(log_q, p, reduction='batchmean')
        return kl_loss
    

    def compute_hinton_loss_multilabel(self, outputs, t_outputs, kd_temp=3):
        # Apply temperature scaling and sigmoid
        probs = F.logsigmoid(outputs / kd_temp)
        t_probs = F.sigmoid(t_outputs / kd_temp)

        # KL divergence (element-wise)
        kd_loss = nn.KLDivLoss(reduction="batchmean")(probs, t_probs) * (kd_temp * kd_temp)
        return kd_loss
    

    def soft_supcon_patch_loss(self, anchor_features, positive_features, distributions, temperature=0.07):
        """
        features: [batch_size, num_patches, feature_dim] (e.g., 32, 196, 768)
        distributions: [batch_size, batch_size] (e.g., soft similarities between samples)
        """
        B, P, D = positive_features.shape  # 32, 196, 768

        # Use CLS token (e.g., first token) as anchor
        anchors = anchor_features  # [B, D]
        anchors = F.normalize(anchors, dim=1)

        # Flatten all non-anchor patch features as positive candidates
        positives = positive_features  # [B, P, D]
        positives = F.normalize(positives, dim=1)

        # Compute dot product between anchor and each of its K patch positives
        logits = torch.bmm(positives, anchors.unsqueeze(-1)).squeeze(-1)  # [B, K]
        logits /= temperature

        # Softmax with soft targets
        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(distributions * log_probs).sum(dim=1).mean()

        return loss

    def anchor_positive_cosine_loss(self, features, mode='mse'):
        """
        features: Tensor of shape (batch_size, 197, feature_dim)
              where [:, 0, :] is the anchor and [:, 1:, :] are the positives
        Returns: scalar tensor loss
        """
        anchors = features[:, 0, :]                      # (32, 768)
        positives = features[:, 1:, :]                   # (32, 196, 768)

        if mode == 'cosine':
            anchors = F.normalize(anchors, dim=1)           # (B, D)
            positives = F.normalize(positives, dim=2)       # (B, P, D)
            sims = torch.bmm(positives, anchors.unsqueeze(2)).squeeze(2)  # (B, P)
            loss = 1 - sims.mean(dim=1)                     # (B,)
        elif mode == 'mse':
            diffs = positives - anchors.unsqueeze(1)        # (B, P, D)
            #loss = (diffs ** 2).mean(dim=(1, 2))            # (B,)
            loss = torch.abs(diffs).mean(dim=(1, 2)) 
            #loss = torch.norm(diffs, dim=2).mean(dim=1)
        else:
            raise ValueError("mode must be 'cosine' or 'mse'") 
        
        return loss.mean()  
    

    def intra_patch_contrastive_loss(self, z, k=10, temperature=0.07):
        """
        z: (B, 196, d_dim) patch embeddings from a batch of images
        k: number of nearest neighbors (positives) per patch
        """
        B, N, D = z.shape

        # Normalize along the feature dimension
        z = F.normalize(z, dim=2)  # (B, N, D)

        # Compute similarity matrices for all B examples in batch
        sim = torch.matmul(z, z.transpose(1, 2))  # (B, N, N)

        # Mask self-similarity
        self_mask = torch.eye(N, device=z.device).unsqueeze(0).bool()  # (1, N, N)
        sim = sim.masked_fill(self_mask, -9e15)

        # Get top-k indices (positives) along dim=2
        #topk = torch.topk(sim, k=k, dim=2)
        #pos_indices = topk.indices  # (B, N, k)

        # Compute logits for softmax
        logits = sim / temperature  # (B, N, N)

        # Create mask for positives
        positive_mask = torch.zeros_like(logits).bool()  # (B, N, N)
        #positive_mask.scatter_(2, pos_indices, True)

        # Numerator: sum of exp over positive logits
        exp_logits = torch.exp(logits)
        numerator = (exp_logits * positive_mask).sum(dim=2)  # (B, N)

        # Denominator: sum over all exp logits
        denominator = exp_logits.sum(dim=2)  # (B, N)

        # Compute loss per patch
        loss = -torch.log(numerator / (denominator + 1e-8))  # (B, N)
        return loss.mean()
    
    def __init__(self, temperature=0.07, lambda1=1, lambda2=1, lambda3=2, num_classes=51):
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
        self.temperature = temperature
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.beta = 0.7  # Weighting factor for cross-entropy loss vs info_nce loss

        # Supervised Contrastive Loss for (Batch, N_Views, Features)
        #self.supconloss = SupConLoss()
        self.inf_nce_loss = SupervisedInfoNCE(temperature=0.07)
        self.orthogonal_loss = OrthogonalLossFunction()  # Adjust num_classes as needed
        self.sup_con = SoftSupConLoss(contrast_mode='one')

    

    def forward(self, sparse_logits, binary_tensor, pool_h, original, reconstructed, logits2, logits, labels):
        """
        Computes the total loss as a weighted sum of entropy loss, sparsity loss, 
        and supervised contrastive loss.
        Args:
            sparse_rep (Tensor): The sparse representation of the input data.
            labels (Tensor): Ground truth labels for supervised contrastive loss.
        Returns:
            Tensor: The total computed loss.
        """
        #return self.weighted_bce_loss(logits, binary_tensor) + self.compute_hinton_loss(logits, reconstructed)
        ce_loss =  nn.CrossEntropyLoss()(logits, labels)
        mse_loss = nn.MSELoss()(reconstructed, original)
        #hinton_loss = self.compute_hinton_loss(logits2, logits)
        #info_nce_loss = self.sup_con(logits2, logits)

        """This is for mse loss"""
        #info_nce_loss = self.anchor_positive_cosine_loss(logits2, mode='mse')

        """This is with kl loss"""
        #info_nce_loss = self.compute_kl_loss(logits2, logits)

        info_nce_loss = 1 - F.cosine_similarity(logits2, pool_h, dim=-1).mean()

        sparse_loss = torch.mean(torch.abs(sparse_logits))
        #return ce_loss + hinton_loss + 0.5 * sparse_loss + mse_loss
        #return ce_loss + sparse_loss + mse_loss
        #print(ce_loss.item(), mse_loss.item(), info_nce_loss.item(), sparse_loss.item())

        # 8e-5, 0.3 -> best one

        # Increased k in intra_patch_contrastive_loss to 10
        #return 0.2 * ce_loss + 0.15 * info_nce_loss + 0.6 * mse_loss + 0.05 * sparse_loss #+ 0.3 * self.intra_patch_contrastive_loss(sparse_logits)
        """This is for abs mse loss"""
        #return self.beta * ce_loss + (1 - self.beta) * info_nce_loss + mse_loss + 4e-5 * sparse_loss
    
        return ce_loss + (1 - self.beta) * info_nce_loss + mse_loss + 4e-5 * sparse_loss
        #return torch.nn.functional.mse_loss(reconstructed, original) + 4e-5 * torch.mean(torch.abs(sparse_rep)) + F.cross_entropy(logits, labels)
        #return self.weighted_bce_loss(sparse_rep, binary_tensor) + self.orthogonal_loss(sparse_rep, segments) #+ self.sup_con_loss(sparse_rep, labels)