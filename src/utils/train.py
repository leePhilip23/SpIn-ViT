import os
import logging
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.dataloader import GTSRB_dataloaders, Flowers102Dataset
from utils.loss_functions import ViTSAELoss
from models.vit_sae import ViTSAE, VanillaVit
from torch.utils.data import DataLoader
from torchvision.datasets import Flowers102
from torchvision import transforms


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
    test_dataloader = DataLoader(data_test, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    """
    

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
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-08)# weight_decay=0.01)
    
    # ViViT Training loop with Early Stopping and Best Model Saving
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3

    logging.info("Training GTSRB")

    # ViViT Training loop
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0
        for i, (image, binary_tensor, segments, label) in enumerate(train_dataloader):
        #for i, (image, labels) in enumerate(train_dataloader):
            image, binary_tensor, labels = image.to(device), binary_tensor.to(device), label.to(device)
            #image, labels = image.to(device), labels.to(device)
            optimizer.zero_grad()
            """
            logits = model(image)
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

            """
            sparse_rep = model(image)
            loss = criterion(
                sparse_rep, 
                binary_tensor, 
                segments,
                labels
            )
            #sparse_rep, orig, reconstructed, y_pred = model(image)
            sparse_logits, logits, original_logit, recon_logit, _ = model(image)
            loss = criterion(
                sparse_logits, 
                binary_tensor, 
                segments,
                original_logit,
                recon_logit,
                logits,
                labels
            )
            loss = criterion(
                sparse_rep, 
                binary_tensor, 
                segments,
                orig,
                reconstructed,
                y_pred,
                labels
            )
            """
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
            for i, (image, binary_tensor, segments, label) in enumerate(val_dataloader):
            #for i, (image, labels) in enumerate(val_dataloader):
                image, binary_tensor, labels = image.to(device), binary_tensor.to(device), label.to(device)
                #image, labels = image.to(device), labels.to(device)
                """
                logits = model(image)
                loss = criterion(logits, labels)
                """
                sparse_logits, original, reconstructed, predicted, logits2, pool_h = model(image)
                loss = criterion(
                    sparse_logits,
                    None,
                    pool_h,
                    original,
                    reconstructed,
                    logits2,
                    predicted,
                    labels
                )
                
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                train_accuracy = correct / total if total > 0 else 0

                #sparse_rep, orig, reconstructed, y_pred = model(image)
                """
                #sparse_logits, logits, original_logit, recon_logit, _ = model(image)
                loss = criterion(
                    sparse_logits, 
                    binary_tensor, 
                    segments,
                    original_logit,
                    recon_logit,
                    logits,
                    labels
                )
                loss = criterion(
                    sparse_rep, 
                    binary_tensor, 
                    segments,
                    orig,
                    reconstructed,
                    y_pred,
                    labels
                )
                """
                running_loss += loss.item()

                # Calculate and print the average loss for the current batch
                avg_loss = running_loss / (i + 1)
                print(f"Epoch {epoch+1}/{args.num_epochs}, Batch {i+1}/{len(val_dataloader)}, Average Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}")
                print("="*50)

        #val_loss = round(running_loss / len(test_dataloader), 3)
        val_loss = running_loss / len(val_dataloader)
        logging.info(f"Validation Loss: {val_loss}, Accuracy: {train_accuracy:.4f}")

        # Early stopping condition
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter when loss improves
            # Save the best model
            best_model_path = os.path.join(args.model_dir, args.dataset_name, "best_model_ce_cs-1.pth")
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