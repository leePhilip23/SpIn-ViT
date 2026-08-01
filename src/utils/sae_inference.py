import os
import copy
import glob
import heapq
import shutil
import torch
import torch.nn as nn
import numpy as np
from datasets import load_from_disk, load_dataset
from models.vit_sae import (
    SpInViT, 
    VanillaSpInViT, 
    PatchSAE, 
    VanillaVit
)
from utils.dataloader import (
    HuggingFaceData,
    FolderData
)
from utils.train import get_dataloader
from torch.utils.data import DataLoader 
from sklearn.model_selection import train_test_split
from PIL import Image
    


def load_vivit_sae_model(args, num_classes):
    """
    Load a pre-trained ViViT model from the given file path.
    Args:
        args (Namespace): Arguments containing the model's state dictionary path.
    Returns:
        dict: Loaded state dictionary.
    """
    state_dict = torch.load(args.vit_sae_model_pth)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpInViT2(num_classes=num_classes, factor=10, topk=True).to(device)
    model.load_state_dict(state_dict)
    return model


def vit_sae_forward(model, image_tensor):
    """
    Perform a forward pass through the ViViT-SAE model in inference mode.
    Args:
        video_tensor (torch.Tensor): Preprocessed video input tensor.
    Returns:
        tuple: (sae_latent, att_features) as CPU tensors.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        _, sparse_logits, _, _ = model(image_tensor)
    return sparse_logits.mean(dim=1).squeeze(0).cpu()


def cache_activations(args, vit_sae, test_dataloader, sample_batch=1000):
    """
    Caches activations for video files by processing frames through ViViT and SAE models.
    Args:
        args (Namespace): Configuration arguments.
        image_processor (callable): Preprocessor for video frames.
        vivit (nn.Module): ViViT model for feature extraction.
        sae (nn.Module): SAE model for activations.
        video_paths (list of str): List of video file paths.
    Returns:
        None
    """
    if os.path.exists(args.activations_dir):
        shutil.rmtree(args.activations_dir)

    img_files = []
    os.makedirs(args.activations_dir, exist_ok=True)

    for i, data in enumerate(test_dataloader):
        file, image, label = data[0], data[1], data[2]
        if sample_batch <= 0:
            break

        img_shape = image.shape

        # Get maximum for each
        for i in range(img_shape[0]):
            print(f"{os.path.splitext(os.path.basename(file[i]))[0]}_activations_{label[i]}.npy")
            output_file = os.path.join(args.activations_dir, f"{os.path.splitext(os.path.basename(file[i]))[0]}_activations_{label[i]}.npy")
            if os.path.exists(output_file):
                continue
            img_files.append(f"{file[i]}")
            image_tensor = copy.deepcopy(image[i]).unsqueeze(0)
            sae_features = vit_sae_forward(vit_sae, image_tensor)

            # Save activations as .npy file
            np.save(output_file, sae_features.numpy())
            #torch.save(image[i], f"{output_file[:-4]}.pt")
        
        sample_batch -= 1

    return img_files


def get_top_k(args):
    """
    Identifies the top-k video paths based on activation values for a given model.
    Args:
        args (Namespace): Contains model paths, activations directory, and output paths.
        image_processor (object): Preprocessor for image data.
        video_paths (list): List of video file paths.
        k (int, optional): Number of top activations to retrieve. Defaults to 68.
    Returns:
        None
    """
    #_, _, test_dataloader, num_classes, _ = GTSRB_dataloaders()
    test_dataloader, num_classes = get_dataloader(True, iteration=0)
    vit_sae = load_vivit_sae_model(args, num_classes=num_classes)
    img_files = cache_activations(args, vit_sae, test_dataloader)

    num_activations = 10 * num_classes
    activation_files = os.listdir(args.activations_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_txt_file, "w") as f:
        for n in range(num_activations):
            top_k, top_k_videos = 5, []
            for activation_file, img_path in zip(sorted(activation_files), sorted(img_files)):
                if not ".npy" in activation_file:
                    continue
                activation_npy_path = os.path.join(args.activations_dir, activation_file)
                activation_neuron = np.load(activation_npy_path).tolist()[n]
                top_k_videos.append((-activation_neuron, img_path))

            # Get top 5 video paths for each activation neuron
            heapq.heapify(top_k_videos)
            while top_k_videos and top_k > 0:
                activation, vid_path = heapq.heappop(top_k_videos)

                if -activation < 0:
                    break

                # Write to txt file
                f.write(f"{n}:{vid_path}\n")
                top_k -= 1
    f.close()