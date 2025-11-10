import os
import copy
import glob
import heapq
import shutil
import torch
import torch.nn as nn
import numpy as np
from datasets import load_from_disk, load_dataset
from models.vit_sae import ViTSAE
from utils.dataloader import (
    GTSRB_dataloaders, 
    PLACES_dataloaders, 
    CatsDogsDataset, 
    Flowers102Dataset, 
    HuggingFaceData,
    FolderData
)
from torch.utils.data import DataLoader 
from sklearn.model_selection import train_test_split
from PIL import Image


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()   # lightweight integrity check
        return True
    except OSError:
        print("Warning")
        return False
    except Exception:
        print(path)
        return False
    


def is_val_image(img: Image) -> bool:
    try:
        img.verify()   # lightweight integrity check
        return True
    except OSError:
        print("Warning")
        return False
    except Exception:
        print("Exception")
        return False
    


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
    #train_dataloader, val_dataloader, test_dataloader, num_classes, class_weights = GTSRB_dataloaders()
    dataset_name = ""
    if dataset_name == "PLACES2":
        train_dataloader, val_dataloader, test_dataloader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["bathroom", "bedroom"],train_shuffle=True)
    if dataset_name == "PLACES3":
        train_dataloader, val_dataloader, test_dataloader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["bathroom", "bedroom", "kitchen"],train_shuffle=True)
    if dataset_name == "PLACES5":
        train_dataloader, val_dataloader, test_dataloader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["bathroom", "bedroom", "kitchen", "dining_room", "living_room"],train_shuffle=True)
    if dataset_name == "PLACES10":
        train_dataloader, val_dataloader, test_dataloader, num_classes, class_weights = PLACES_dataloaders(
            class_list=["bathroom", "bedroom", "kitchen", "dining_room", "living_room", "home_office", "office",
                        "waiting_room", "conference_room", "hotel_room"],train_shuffle=True)
        

    """
    files = [
        f for f in glob.glob("data/raw/pets/**/*", recursive=True)
        if os.path.isfile(f) 
        and "db" not in f
        and is_valid_image(f)
    ]
    
    labels = [path.split('/')[-2].split('.')[0] for path in files]
    """

    """
    data = load_from_disk("data/raw/cats_vs_dogs")
    files, labels = [], []
    for d in data:
        img, label = d['image'], d['labels']
        if not is_val_image(img):
            continue
        
        files.append(img)
        labels.append(label)
    """
    """
    X_train, X_test, y_train, y_test = train_test_split(
        files,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=1
    )

    train_data = None
    val_data = CatsDogsDataset(X_test, y_test, inference=True)
    train_dataloader = None
    test_dataloader = DataLoader(val_data, batch_size=8, shuffle=False, num_workers=0, pin_memory=True)
    val_dataloader, num_classes, class_weights = None, 2, None
    
    """

    #num_classes, class_weights = 102, None
    #data_train = Flowers102Dataset("train")
    #data_val = Flowers102Dataset("valid")
    #data_test = Flowers102Dataset("valid", inference=True)

    #num_classes, class_weights = 102, None
    #data_train = Flowers102Dataset("train")
    #data_val = Flowers102Dataset("valid")

    #num_classes, class_weights = 196, None
    #data_test = FolderData("data/raw/cars/test", inference=True)

    #num_classes, class_weights = 100, None
    #data_test = FolderData("data/raw/fgvc/test", inference=True, csv="data/raw/fgvc/test.csv")

    #num_classes, class_weights = 102, None
    #data = load_dataset(
    #    "HuggingFaceM4/Caltech-101",
    #    "with_background_category"
    #).shuffle(seed=1)
    
    #data_test = HuggingFaceData(data, split='test', inference=True)

    """
    num_classes, class_weights = 47, None
    data = load_dataset("tanganke/dtd")
    data_train = HuggingFaceData(data, split='train')
    data_val = HuggingFaceData(data, split='test')
    """
    num_classes, class_weights = 10, None
    data = load_dataset("timm/eurosat-rgb")
    data_test = HuggingFaceData(data, split='validation')
    #"""

    train_dataloader = None
    val_dataloader = None
    test_dataloader = DataLoader(data_test, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

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

    num_activations = 768 * 64
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