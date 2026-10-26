import torch
import torch.nn as nn
from transformers import ViTModel


class SpInViT(nn.Module):
    """ViT backbone with a sparse autoencoder over all token features.

    Returns (vit_features, sparse_logits, recon_logit, cls_logits).
    """

    def __init__(
        self,
        num_classes,
        d_dim=768,
        factor=10,
        output_attentions=False,
        topk=False
    ):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.k = num_classes * 5
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=output_attentions)

        self.encoder = nn.Linear(d_dim, num_classes * factor)
        self.decoder = nn.Linear(num_classes * factor, d_dim)
        self.linear_out = nn.Linear(d_dim, num_classes)
        self.sigmoid = nn.Sigmoid()

    def _keep_topk(self, x):
        """Zero all but the k largest activations along the last dim."""
        topk_vals, indices = torch.topk(x, self.k, dim=-1)
        return torch.zeros_like(x).scatter(-1, indices, topk_vals)

    def forward(self, x):
        out = self.vit(x)
        cls_logits = self.linear_out(out.pooler_output)

        vit_features = out.last_hidden_state
        sparse_logits = self.sigmoid(self.encoder(vit_features))
        if self.topk:
            sparse_logits = self._keep_topk(sparse_logits)
        recon_logit = self.decoder(sparse_logits)

        return (
            vit_features,
            sparse_logits,
            recon_logit,
            cls_logits
        )
