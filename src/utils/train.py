import os
import logging
import torch

import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from utils.dataloader import (
    HuggingFaceData,
    FolderData
)
from utils.loss_functions import ViTSAELoss
from models.vit_sae import (
    SpInViT, 
    VanillaSpInViT, 
    PatchSAE, 
    VanillaVit
)
from torch.utils.data import DataLoader
import math
from scipy.stats import norm
from transformers import set_seed

# Set the global seed for reproducibility
set_seed(42)


def moe(correct, total, confidence=0.95):
    p_hat = correct / total
    z = norm.ppf(1 - (1 - confidence) / 2)
    moe = z * math.sqrt((p_hat * (1 - p_hat)) / total)
    return moe
    

def get_dataloader(inference: bool, iteration: int=0, train_batch: int=32, val_batch: int=16):
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

    if iteration == 0:
        num_classes, class_weights = 102, None
        data = load_dataset("pufanyi/flowers102")
        data_train = HuggingFaceData(data, split='train', too_big=True, inference=inference)
        data_val = HuggingFaceData(data, split='test', too_big=True, inference=inference)
        #data_val = HuggingFaceData(data, split='validation', too_big=True, inference=inference)
    elif iteration == 1:
        num_classes, class_weights = 196, None
        data_train = FolderData("data/raw/cars/train", inference=inference)
        data_val = FolderData("data/raw/cars/test", inference=inference)
    elif iteration == 2:
        num_classes, class_weights = 100, None
        data_train = FolderData("data/raw/fgvc/train", csv="data/raw/fgvc/train.csv", inference=inference)
        data_val = FolderData("data/raw/fgvc/train", csv="data/raw/fgvc/val.csv", inference=inference)
    elif iteration == 3:
        num_classes, class_weights = 102, None
        data = load_dataset(
            "HuggingFaceM4/Caltech-101",
            "with_background_category"
        ).shuffle(seed=42)
        data_train = HuggingFaceData(data, split='train', too_big=True, inference=inference)
        data_val = HuggingFaceData(data, split='test', too_big=True, inference=inference)
    elif iteration == 4:
        # Food101
        num_classes, class_weights = 101, None
        data = load_dataset("ethz/food101")
        data_train = HuggingFaceData(data, split='train', too_big=True, inference=inference)
        data_val = HuggingFaceData(data, split='validation', too_big=True, inference=inference)
    elif iteration == 5:
        #Sun397
        num_classes, class_weights = 397, None
        data = load_dataset("1aurent/SUN397")['train'].train_test_split(
        test_size=0.2,
            stratify_by_column='label',
            seed=42
        ) 
        data_train = HuggingFaceData(data, split='train', too_big=True, inference=inference)
        data_val = HuggingFaceData(data, split='test', too_big=True, inference=inference)
    elif iteration == 6:
        num_classes, class_weights = 47, None
        data = load_dataset("tanganke/dtd")
        data_train = HuggingFaceData(data, split='train', too_big=True, inference=inference)
        data_val = HuggingFaceData(data, split='test', too_big=True, inference=inference)
    elif iteration == 7:
        num_classes, class_weights = 10, None
        data = load_dataset("timm/eurosat-rgb")
        data_train = HuggingFaceData(data, split='train', too_big=True, inference=inference)
        data_val = HuggingFaceData(data, split='validation', too_big=True, inference=inference)
    elif iteration == 8:
        num_classes, class_weights = 37, None
        data = load_dataset("timm/oxford-iiit-pet")
        data_train = HuggingFaceData(data, split='train', inference=inference)
        data_val = HuggingFaceData(data, split='test', inference=inference)


    print(f"Num Classes: {num_classes}")
    if not inference:
        train_dataloader = DataLoader(data_train, batch_size=train_batch, shuffle=True, num_workers=4, pin_memory=True)

    val_dataloader = DataLoader(data_val, batch_size=val_batch, shuffle=True, num_workers=4, pin_memory=True)
    return (
        train_dataloader, 
        val_dataloader, 
        None, 
        num_classes, 
        class_weights
    ) if not inference else (
        val_dataloader, 
        num_classes
    )


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

    #"""
    model = SpInViT(
        num_classes, 
        factor=10, 
        topk=True,
        separate=False
    ).to(device)
    criterion = ViTSAELoss(num_cls=num_classes)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=1e-4, 
        betas=(0.9, 0.999), 
        eps=1e-08, 
        weight_decay=0.01
    )

    # Training loop with Early Stopping and Best Model Saving
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3

    logging.info(f"Training {args.dataset_name} (Proposed)")

    #  Training loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss, total = 0, 0
        for i, (image, labels) in enumerate(train_dataloader):
            image, labels = image.to(device), labels.to(device)
            optimizer.zero_grad()
            """
            logits = model(image).squeeze()
            loss = criterion(logits, labels)
            """
            vit_features, sparse_logits, recon_logit, cls_logits = model(image)
            loss = criterion(
                vit_features,
                sparse_logits,
                recon_logit,
                cls_logits,
                labels,
                num_classes
            )
            #"""
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate and print the average loss for the current batch
            avg_loss = running_loss / (i + 1)
            print(f"Epoch {epoch+1}/{args.num_epochs}, Batch {i+1}/{len(train_dataloader)}, Average Loss: {avg_loss:.4f}")
            print("="*50)

        train_loss = running_loss / len(train_dataloader)
        logging.info(f"Training Loss: {train_loss}")


        model.eval()
        running_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for i, (image, labels) in enumerate(val_dataloader):
                image, labels = image.to(device), labels.to(device)
                """
                cls_logits = model(image).squeeze()
                loss = criterion(cls_logits, labels)
                """
                vit_features, sparse_logits, recon_logit, cls_logits = model(image)
                loss = criterion(
                    vit_features,
                    sparse_logits,
                    recon_logit,
                    cls_logits,
                    labels,
                    num_classes
                )
                #"""
                
                _, predicted = torch.max(cls_logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                train_accuracy = correct / total if total > 0 else 0
                running_loss += loss.item()

                # Calculate and print the average loss for the current batch
                avg_loss = running_loss / (i + 1)
                print(f"Epoch {epoch+1}/{args.num_epochs}, Batch {i+1}/{len(val_dataloader)}, Average Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")
                print("="*50)

        val_loss = running_loss / len(val_dataloader)
        logging.info(f"Validation Loss: {val_loss}, Accuracy: {(train_accuracy * 100.0):.5f} += {(moe(correct, total) * 100.0):.3f}")

        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter when loss improves
            # Save the best model
            best_model_path = os.path.join(args.model_dir, args.dataset_name, "best_topk_model.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Epoch {epoch+1}: Validation loss improved, saving best model.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                logging.info("\n")
                break

        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(args.model_dir, args.dataset_name, f"sae_{epoch+1}.pth"))