import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel

"""
class ViTSAE(nn.Module):
    def __init__(self, num_classes, d_dim=768, output_attentions=False):
        super(ViTSAE, self).__init__()
        self.num_classes = num_classes
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=output_attentions)
        self.encoder = nn.Linear(d_dim, num_classes * 5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vit(x)["last_hidden_state"][:, 1:]
        x = self.encoder(x)
        return self.sigmoid(x.mean(dim=1))
    

class ViTSAE(nn.Module):
    def __init__(self, num_classes, d_dim=768, output_attentions=False):
        super(ViTSAE, self).__init__()
        self.num_classes = num_classes
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=output_attentions)
        self.encoder = nn.Linear(d_dim, num_classes * 5)
        self.decoder = nn.Linear(num_classes * 5, d_dim)
        self.linear_out = nn.Linear(768, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.vit(x)
        y_pred = self.linear_out(x.pooler_output)
        # TODO: Either do avg pooling before encoder or after it
        original = x.last_hidden_state[:, 1:].mean(dim=1)
        x = self.encoder(original)
        sparse_rep = self.relu(x)
        reconstructed = self.decoder(sparse_rep)
        return sparse_rep, original, reconstructed, y_pred
"""
class Prune(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, embed_dim=768):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU()
        )

        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, policy):
        x = self.in_conv(x)
        B, N, C = x.size()
        local_x = x[:,:, :C//2]
        global_x = (x[:,:, C//2:] * policy).sum(dim=1, keepdim=True) / torch.sum(policy, dim=1, keepdim=True)
        x = torch.cat([local_x, global_x.expand(B, N, C//2)], dim=-1)
        return self.out_conv(x)
    

class ViTSAE(nn.Module):
    def __init__(self, num_classes, d_dim=768, output_attentions=False):
        super(ViTSAE, self).__init__()
        self.num_classes = num_classes
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=output_attentions)
        self.query = nn.Linear(768, 768)
        self.key = nn.Linear(768, 768)
        self.value = nn.Linear(768, 768)
        self.fc1 = nn.Linear(768, 1)
        self.fc2 = nn.Linear(196, num_classes)
        #self.encoder = nn.Linear(d_dim, num_classes * 5)
        #self.encoder = nn.Linear(196, num_classes * 5)
        self.encoder = nn.Linear(768, 768*16)
        self.decoder = nn.Linear(768*16, 768)
        #self.linear_out = nn.Linear(768, num_classes * 5)
        self.linear_out = nn.Linear(768, num_classes)
        self.linear_out2 = nn.Linear(768, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        self.prune = Prune()

    def forward(self, x):
        x = self.vit(x)
        pool_h = x.pooler_output
        logits = self.linear_out(pool_h)
        #class_rep = self.sigmoid(self.linear_out(pool_h))

        h = x.last_hidden_state[:, 1:]
        #h = self.prune(h, torch.ones(x.shape[0], 196, 768))
        '''
        mixed_query_layer = self.query(h)
        mixed_key_layer = self.key(h)
        mixed_value_layer = self.value(h)
        
        attention_scores = torch.matmul(mixed_query_layer, mixed_key_layer.transpose(-1, -2))
        attention_probs = self.softmax(attention_scores)
        
        context_layer = torch.matmul(attention_probs, mixed_value_layer)
        new_atts = attention_probs
        inter_logits = self.fc2(self.fc1(context_layer).squeeze(dim = 2))
        sparse_logits = self.relu(self.encoder(inter_logits))
        '''
        sparse_logits = self.relu(self.encoder(h))
        recon_logit = self.decoder(sparse_logits)

        #cls_token = pool_h.unsqueeze(1)  # (32, 1, 768)

        """This is for mse absolute loss"""
        #whole_output = torch.cat((cls_token, recon_logit), dim=1)

        # For knowledge distillation
        #logits2 = self.linear_out2(recon_logit.mean(dim=1))

        # Just get features
        logits2 = recon_logit.mean(dim=1)


        # For info nce
        #logits2 = self.linear_out2(recon_logit)
        #sparse_rep = self.sigmoid(sparse_logits)

        #return sparse_rep, logits, new_atts
        #return sparse_logits, logits, inter_logits, recon_logit, new_atts

        #return sparse_logits, h, recon_logit, logits, logits2
        """This is for mse absolute loss"""
        #return sparse_logits, h, recon_logit, logits, whole_output #logits2

        """This is for kl div"""
        return sparse_logits, h, recon_logit, logits, logits2, pool_h
    
    
class VanillaVit(nn.Module):
    def __init__(self, num_classes, d_dim=768, output_attentions=False):
        super(VanillaVit, self).__init__()
        self.num_classes = num_classes
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=output_attentions)
        self.linear_out = nn.Linear(d_dim, num_classes)

    def forward(self, x):
        x = self.vit(x)
        pool_h = x.pooler_output
        return self.linear_out(pool_h)