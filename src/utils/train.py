import os
import glob
import logging
import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_from_disk, load_dataset
from utils.dataloader import (
    GTSRB_dataloaders, 
    PLACES_dataloaders, 
    CatsDogsDataset, 
    Flowers102Dataset, 
    HuggingFaceData,
    FolderData
)
from utils.loss_functions import ViTSAELoss
from models.vit_sae import ViTSAE, VanillaVit
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError
import warnings
import math
from scipy.stats import norm

# Convert all warnings to errors
warnings.filterwarnings("error")

def moe(correct, total, confidence=0.95):
    p_hat = correct / total
    z = norm.ppf(1 - (1 - confidence) / 2)
    moe = z * math.sqrt((p_hat * (1 - p_hat)) / total)
    return moe


def is_valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()   # lightweight integrity check
            img.close()
        return True
    except (OSError, IOError, UnidentifiedImageError, Exception):
        print(path)
        img.close()
        os.remove(path)
        return False
    

def is_val_image(img: Image) -> bool:
    try:
        img.verify()   # lightweight integrity check
        img.close()
        return True
    except Exception:
        img.close()
        print("Exception")
        return False
    

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
        

    """
    files = [
        f for f in glob.glob("data/raw/pets/**/*", recursive=True)
        if os.path.isfile(f) 
        and "db" not in f
        and is_valid_image(f)
    ]
    
    labels = [path.split('/')[-2].split('.')[0] for path in files]

    print('Dog' in labels and 'Cat' in labels)
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
    train, val = train_test_split(
        files,
        test_size=0.2,
        stratify=labels,
        random_state=1
    )


    train_data = CatsDogsDataset(train, transform="train")
    val_data = CatsDogsDataset(val)
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader, num_classes, class_weights = None, 2, None
    
    """
    #num_classes, class_weights = 102, None
    #data_train = Flowers102Dataset("train")
    #data_val = Flowers102Dataset("valid")

    #num_classes, class_weights = 196, None
    #data_train = FolderData("data/raw/cars/train")
    #data_val = FolderData("data/raw/cars/test")

    #num_classes, class_weights = 100, None
    #data_train = FolderData("data/raw/fgvc/train", csv="data/raw/fgvc/train.csv")
    #data_val = FolderData("data/raw/fgvc/train", csv="data/raw/fgvc/val.csv")

    #num_classes, class_weights = 102, None
    #data = load_dataset(
    #    "HuggingFaceM4/Caltech-101",
    #    "with_background_category"
    #).shuffle(seed=42)
    #data_train = HuggingFaceData(data, split='train')
    #data_val = HuggingFaceData(data, split='test')


    #Food101
    #num_classes, class_weights = 101, None
    #data = load_dataset("ethz/food101")
    #data_train = HuggingFaceData(data, split='train', too_big=True)
    #data_val = HuggingFaceData(data, split='validation', too_big=True)

    #Sun397
    #num_classes, class_weights = 397, None
    #data = load_dataset("1aurent/SUN397")['train'].train_test_split(
    #   test_size=0.2,
    #    stratify_by_column='label',
    #    seed=42
    #)
    #data_train = HuggingFaceData(data, split='train', too_big=True)
    #data_val = HuggingFaceData(data, split='test', too_big=True)
    #data_train = FolderData("data/raw/suns397/train")
    #data_val = FolderData("data/raw/suns397/test")
    
    #num_classes, class_weights = 37, None
    #data = load_dataset("enterprise-explorers/oxford-pets")
    # Train need to split

    #data_train = HuggingFaceData(data, split='train')
    #data_val = HuggingFaceData(data, split='validation')


    #num_classes, class_weights = 47, None
    #data = load_dataset("tanganke/dtd")
    #data_train = HuggingFaceData(data, split='train', too_big=True)
    #data_val = HuggingFaceData(data, split='test', too_big=True)
    #"""
    #num_classes, class_weights = 10, None
    #data = load_dataset("timm/eurosat-rgb")
    #data_train = HuggingFaceData(data, split='train')
    #data_val = HuggingFaceData(data, split='validation')
    #"""

    num_classes, class_weights = 37, None
    data = load_dataset("timm/oxford-iiit-pet")
    data_train = HuggingFaceData(data, split='train')
    data_val = HuggingFaceData(data, split='test')

    # FGVC batch size 8
    train_dataloader = DataLoader(data_train, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(data_val, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = None
    
    print(f"Num Classes: {num_classes}")
    return train_dataloader, val_dataloader, test_dataloader, num_classes, class_weights


def train_model(args, device, train_dataloader, val_dataloader, num_classes):
    """
    Trains the ViViTAndSparseEncoder model.
    Args:
        train_dataloader (DataLoader): Training data loader.
        test_dataloader (DataLoader): Validation data loader.
        device (torch.device): Device (CPU/GPU).
        args (Namespace): Training parameters.
    Steps:
    1. Initialize model, loss, and optimizer.
    2. Train and validate per epoch.
    3. Save the best model and apply early stopping.
    4. Save checkpoints every 5 epochs.
    """

    model = ViTSAE(num_classes).to(device)
    criterion = ViTSAELoss()

    #model = VanillaVit(num_classes).to(device)
    #criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCEWithLogitsLoss()

    # Stanford Cars (1e-4), FGVC (2e-4) Batch 32
    #optimizer = optim.AdamW(model.parameters(), lr=2e-4)
    # SUNS397 Food101 (batch 32)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)#, betas=(0.9, 0.999), eps=1e-08)# weight_decay=0.01)
    #optimizer = optim.AdamW(model.parameters(), lr=1e-4)#, betas=(0.9, 0.999), eps=1e-08)
    
    # ViViT Training loop with Early Stopping and Best Model Saving
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3

    logging.info(f"Training {args.dataset_name} (Baseline)")

    # ViViT Training loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0
        #for i, (image, binary_tensor, segments, label) in enumerate(train_dataloader):
        for i, (image, labels) in enumerate(train_dataloader):
            #image, binary_tensor, labels = image.to(device), binary_tensor.to(device), label.to(device)
            image, labels = image.to(device), labels.to(device)
            optimizer.zero_grad()
            """
            logits = model(image).squeeze()
            loss = criterion(logits, labels)
            """
            sparse_logits, original, reconstructed, logits, logits2, pool_h = model(image)
            loss = criterion(
                sparse_logits,
                None,
                pool_h,
                original,
                reconstructed,
                logits2,
                logits,
                labels
            )
            #"""
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate and print the average loss for the current batch
            avg_loss = running_loss / (i + 1) 
            print(f"Epoch {epoch+1}/{args.num_epochs}, Batch {i+1}/{len(train_dataloader)}, Average Loss: {avg_loss:.4f}")
            print("="*50)

        #train_loss = round(running_loss / len(train_dataloader), 3)
        train_loss = running_loss / len(train_dataloader)
        logging.info(f"Training Loss: {train_loss}")


        model.eval()
        running_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            #for i, (image, binary_tensor, segments, label) in enumerate(val_dataloader):
            for i, (image, labels) in enumerate(val_dataloader):
                #image, binary_tensor, labels = image.to(device), binary_tensor.to(device), label.to(device)
                image, labels = image.to(device), labels.to(device)
                """
                logits = model(image).squeeze()
                loss = criterion(logits, labels)
                """
                sparse_logits, original, reconstructed, logits, logits2, pool_h = model(image)
                loss = criterion(
                    sparse_logits,
                    None,
                    pool_h,
                    original,
                    reconstructed,
                    logits2,
                    logits,
                    labels
                )
                #"""
                
                _, predicted = torch.max(logits, 1)
                #predicted = F.sigmoid(torch.squeeze(logits))
                #predicted = (predicted >= 0.5)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                train_accuracy = correct / total if total > 0 else 0
                running_loss += loss.item()

                # Calculate and print the average loss for the current batch
                avg_loss = running_loss / (i + 1)
                print(f"Epoch {epoch+1}/{args.num_epochs}, Batch {i+1}/{len(val_dataloader)}, Average Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")
                print("="*50)

        #val_loss = round(running_loss / len(test_dataloader), 3)
        val_loss = running_loss / len(val_dataloader)
        logging.info(f"Validation Loss: {val_loss}, Accuracy: {train_accuracy:.5f} += {moe(correct, total):.6f}")

        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter when loss improves
            # Save the best model
            best_model_path = os.path.join(args.model_dir, args.dataset_name, "best_model_pets.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}: Validation loss improved, saving best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                logging.info("\n")
                break

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, args.dataset_name, f"vivit_sae_{epoch+1}.pth"))