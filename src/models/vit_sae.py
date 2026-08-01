import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, CLIPVisionModel


class ViTSAE(nn.Module):
    def __init__(self, num_classes, d_dim=768, factor=10, output_attentions=False):
        super(ViTSAE, self).__init__()
        self.num_classes = num_classes
        self.k = num_classes * 5
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=output_attentions)
        for param in self.vit.parameters():
            param.requires_grad = False

        self.encoder = nn.Linear(d_dim, num_classes * factor)
        self.decoder = nn.Linear(num_classes * factor, d_dim)
        self.linear_out = nn.Linear(d_dim, num_classes)
        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()

    def forward(self, x):
        vit_features = self.vit(x)
        #y_pred = self.linear_out(x.pooler_output)
        # TODO: Either do avg pooling before encoder or after it
        original = vit_features.last_hidden_state[:, 1:]
        x = self.encoder(original)


        sparse_logits = self.sigmoid(x)
        topk_vals, indices = torch.topk(sparse_logits, self.k, dim=-1)
    
        # Create sparse tensor/mask for the output
        zeros = torch.zeros_like(sparse_logits)
        sparse_logits = zeros.scatter(-1, indices, topk_vals)

        recon_logits = self.decoder(sparse_logits)
        cls_logits = self.linear_out(recon_logits.mean(dim=1))
        return (
            original,
            sparse_logits,
            recon_logits, 
            cls_logits
        )
    
class VanillaSpInViT(nn.Module):
    def __init__(
        self, 
        num_classes,
        d_dim=768, 
        factor=10,  
        output_attentions=False, 
        topk=False, 
        separate=False, 
        d_name='fgvc',
        relative_path='./'
    ):
        super(VanillaSpInViT, self).__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.k = num_classes * 5

        
        if separate:
            self.vit = VanillaVit(num_classes, output_attentions=output_attentions, feats=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(f"{relative_path}model-vitsae/{d_name}/base.pth", map_location=device)
            self.vit.load_state_dict(state_dict)
            self.vit.linear_out = nn.Identity()

            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            print('VIT BASE')
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=output_attentions)

        self.encoder = nn.Linear(768, num_classes * factor)
        self.decoder = nn.Linear(num_classes * factor, 768)
        self.linear_out = nn.Linear(768, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vit(x)
        pool_h = x.pooler_output
        cls_logits = self.linear_out(pool_h)

        vit_features = x.last_hidden_state[:, 1:]
        sparse_logits = self.sigmoid(self.encoder(vit_features))
        recon_logit = self.decoder(sparse_logits)

        return (
            vit_features,
            sparse_logits,
            recon_logit, 
            cls_logits
        )
    

class SpInViT(nn.Module):
    def __init__(
        self, 
        num_classes,
        d_dim=768, 
        factor=10,  
        output_attentions=False, 
        topk=False, 
        separate=False, 
        d_name='fgvc',
        relative_path='./'
    ):
        super(SpInViT, self).__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.k = num_classes * 5

        
        if separate:
            self.vit = VanillaVit(num_classes, output_attentions=output_attentions, feats=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(f"{relative_path}model-vitsae/{d_name}/base.pth", map_location=device)
            self.vit.load_state_dict(state_dict)
            self.vit.linear_out = nn.Identity()

            for param in self.vit.parameters():
                param.requires_grad = False
        else:
            print('VIT BASE')
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=output_attentions)

        self.encoder = nn.Linear(768, num_classes * factor)
        self.decoder = nn.Linear(num_classes * factor, 768)
        self.linear_out = nn.Linear(768, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.vit(x)
        pool_h = x.pooler_output
        cls_logits = self.linear_out(pool_h)
        vit_features = x.last_hidden_state
        sparse_logits = self.sigmoid(self.encoder(vit_features))
        if self.topk:
            topk_vals, indices = torch.topk(sparse_logits, self.k, dim=-1)
        
            # Create sparse tensor/mask for the output
            zeros = torch.zeros_like(sparse_logits)
            sparse_logits = zeros.scatter(-1, indices, topk_vals)
        recon_logit = self.decoder(sparse_logits)

        return (
            vit_features,
            sparse_logits,
            recon_logit, 
            cls_logits
        )
    

class PatchSAE(nn.Module):
    def __init__(self, num_classes, factor=64, d_dim=768, output_attentions=False):
        super(PatchSAE, self).__init__()
        self.num_classes = num_classes
        self.vit = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")

        for param in self.vit.parameters():
            param.requires_grad = False

        self.relu = nn.ReLU()
        self.encoder = nn.Linear(768, 768*64)
        self.decoder = nn.Linear(768*64, 768)
        self.linear_out = nn.Linear(768, num_classes * factor)
        self.linear_out2 = nn.Linear(768, num_classes * factor)

    def forward(self, x):
        x = self.vit(x)
        pool_h = x.last_hidden_state[:, 0]
        cls_logits = self.linear_out(pool_h)
        #class_rep = self.sigmoid(self.linear_out(pool_h))

        vit_features = x.last_hidden_state
        sparse_logits = self.relu(self.encoder(vit_features))
        recon_logit = self.decoder(sparse_logits)

        return (
            vit_features,
            sparse_logits,
            recon_logit, 
            cls_logits
        )
    
    
class VanillaVit(nn.Module):
    def __init__(self, num_classes, d_dim=768, output_attentions=False, feats=False):
        super(VanillaVit, self).__init__()
        self.num_classes = num_classes
        self.feats = feats
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", output_attentions=output_attentions)
        #self.vit = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.linear_out = nn.Linear(d_dim, num_classes)

    def forward(self, x):
        x = self.vit(x)
        if not self.feats:
            pool_h = x.pooler_output
            return self.linear_out(pool_h)
        return x