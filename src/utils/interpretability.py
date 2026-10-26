import os
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import ViTImageProcessor
from utils.sae_inference import load_vivit_sae_model


matplotlib.use("Agg")

PATCH_SIZE = 16
NUM_PATCHES = (224 // PATCH_SIZE) ** 2  # 196
THRESHOLD = 0.5  # SAE activation threshold for a patch to count as active
HIGHLIGHT_COLOR = (1.0, 1.0, 0.3)
HIGHLIGHT_ALPHA = 0.7


def highlight_patches(image_tensor, active):
    """
    Blend a highlight color into every 16x16 patch where `active` is 1.
    Args:
        image_tensor (torch.Tensor): (1, 3, 224, 224) preprocessed image.
        active (torch.Tensor): (196,) binary mask over patches.
    Returns:
        np.ndarray: (224, 224, 3) highlighted image clipped to [0, 1].
    """
    # (1, 3*16*16, 196) -> (196, 3, 16, 16)
    patches = F.unfold(image_tensor, kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    patches = patches.squeeze(0).transpose(0, 1).reshape(NUM_PATCHES, 3, PATCH_SIZE, PATCH_SIZE)

    color = torch.tensor(HIGHLIGHT_COLOR).view(1, 3, 1, 1)
    patches = patches + active[:, None, None, None] * (color - patches) * HIGHLIGHT_ALPHA

    # Fold back to (1, 3, 224, 224)
    patches = patches.reshape(NUM_PATCHES, -1).transpose(0, 1).unsqueeze(0)
    image = F.fold(patches, output_size=(224, 224), kernel_size=PATCH_SIZE, stride=PATCH_SIZE)
    return image.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)


def save_patch_figures(args, num_classes=102, max_images=None):
    """
    For each neuron:image pair in the top-k neurons file, highlight the patches
    where that SpInViT SAE neuron fires and save the figure to
    <interp_dir>/<dataset>/neuron_<idx>_<line>.png.
    """
    data_name = "flowers102" if args.dataset_name == "any" else args.dataset_name
    out_dir = os.path.join(args.interp_dir, data_name)
    os.makedirs(out_dir, exist_ok=True)

    data = load_dataset("pufanyi/flowers102")
    process = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224", use_fast=True)
    model = load_vivit_sae_model(args, num_classes=num_classes)
    model.eval()
    device = next(model.parameters()).device

    saved = 0
    with open(args.output_txt_file) as f:
        for i, line in enumerate(f):
            neuron_idx, img_idx = map(int, line.strip().split(":"))
            image = data["test"][img_idx]["image"]
            try:
                image_tensor = process(images=image, return_tensors="pt").pixel_values
            except Exception:
                continue

            with torch.no_grad():
                _, sparse_logits, _, _ = model(image_tensor.to(device))

            # Patch tokens only (drop CLS), binarized
            active = (sparse_logits[0, 1:, neuron_idx].cpu() > THRESHOLD).float()
            if not active.any():
                continue

            orig = image_tensor.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
            panels = [(orig, "Original"), (highlight_patches(image_tensor, active), "SpInViT")]

            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            for ax, (img, title) in zip(axes, panels):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"neuron_{neuron_idx}_{i}.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"{i} neuron {neuron_idx}")

            saved += 1
            if max_images is not None and saved >= max_images:
                break
