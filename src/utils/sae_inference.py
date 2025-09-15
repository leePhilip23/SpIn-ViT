import os
import copy
import heapq
import shutil
import torch
import torch.nn as nn
import numpy as np
from models.vit_sae import ViTSAE
from utils.dataloader import GTSRB_dataloaders, Flowers102Dataset
from torch.utils.data import DataLoader 


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
    model = ViTSAE(num_classes=num_classes).to(device)
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
        sparse_rep, _, _, _, _, _ = model(image_tensor)
    return sparse_rep.mean(dim=1).squeeze(0).cpu()


def cache_activations(args, vit_sae, test_dataloader, sample_batch=2):
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

    for i, (file, image, label) in enumerate(test_dataloader):
        if sample_batch <= 0:
            break

        img_shape = image.shape

        # Get maximum for each
        for i in range(img_shape[0]):
            print(f"{os.path.splitext(os.path.basename(file[i]))[0]}_activations_{label[i]}.npy")
            output_file = os.path.join(args.activations_dir, f"{os.path.splitext(os.path.basename(file[i]))[0]}_activations_{label[i]}.npy")
            if os.path.exists(output_file):
                continue
            img_files.append(f"{file[i]}.ppm")
            image_tensor = copy.deepcopy(image[i]).unsqueeze(0)
            sae_features = vit_sae_forward(vit_sae, image_tensor)

            # Save activations as .npy file
            np.save(output_file, sae_features.numpy())
        
        sample_batch -= 1

    return img_files



def get_dataloader():
    """
    Creates and returns dataloaders for training and testing datasets.
    Args:
        args: An object containing the following attributes:
            - train_videos_dir (str): Directory path for training videos.
            - train_labels_dir (str): Directory path for training labels.
            - test_videos_dir (str): Directory path for testing videos.
            - test_labels_dir (str): Directory path for testing labels.
    Returns:
        tuple: A tuple containing:
            - train_dataloader (DataLoader): DataLoader for the training dataset.
            - test_dataloader (DataLoader): DataLoader for the testing dataset.
    """
    
    # Load the training dataset
    train_dataloader, val_dataloader, test_dataloader, num_classes, class_weights = GTSRB_dataloaders()
    
    """
    num_classes, class_weights = 102, None
    data_train = Flowers102Dataset("train")
    data_val = Flowers102Dataset("val")
    data_test = Flowers102Dataset("test")

    train_dataloader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(data_val, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(data_test, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    """

    return train_dataloader, val_dataloader, test_dataloader, num_classes, class_weights


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
    _, _, test_dataloader, num_classes, _ = get_dataloader()
    vit_sae = load_vivit_sae_model(args, num_classes)
    img_files = cache_activations(args, vit_sae, test_dataloader)

    num_activations = 768 * 16
    activation_files = os.listdir(args.activations_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(args.output_txt_file, "w") as f:
        for n in range(num_activations):
            top_k, top_k_videos = 5, []
            for activation_file, img_path in zip(sorted(activation_files), sorted(img_files)):
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