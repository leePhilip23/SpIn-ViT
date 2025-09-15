import os
import argparse
import logging
import torch
from utils.train import get_dataloader, train_model
from utils.sae_inference import get_top_k


def main(args):
    if args.train == "true":
        # Create directories for model and log storage
        print("Training the model...")
        os.makedirs(args.model_dir, exist_ok=True)
        os.makedirs(os.path.join(args.model_dir, args.dataset_name), exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)

        # Setup logging credentials and record
        logging_path = os.path.join(args.log_dir, "log.txt")
        logging.basicConfig(filename=logging_path, level=logging.INFO, format='%(message)s')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        train_dataloader, val_dataloader, _, num_classes, _ = get_dataloader()
        train_model(args, device, train_dataloader, val_dataloader, num_classes)
    else:
        print("Getting top-k neurons...")
        get_top_k(args)
    

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a ViT model")
    parser.add_argument("--train", type=str, default="true", help="Whether to train the model or not")

    # Train parameters
    parser.add_argument("--dataset_name", type=str, default="gtsrb", help="The dataset used")
    parser.add_argument("--model_dir", type=str, default="model-vitsae", help="The directory where the model will be saved.")
    parser.add_argument("--log_dir", type=str, default="record-vitsae", help="The directory where the log will be saved.")
    parser.add_argument("--num_epochs", type=int, default=50, help="The number of epochs to train the model.")

    # Inference parameters
    parser.add_argument(
        "--vit_sae_model_pth",
        type=str,
        default=os.path.join("model-vitsae", "gtsrb", "best_model_ce_cs-1.pth"),
        help="Path to the ViViT model file.",
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default=os.path.join("activations", "gtsrb"),
        help="Directory to save the activations files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("record-vitsae", "gtsrb"),
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--output_txt_file",
        type=str,
        default=os.path.join("record-vitsae", "gtsrb", "top_k_neurons.txt"),
        help="Path to the output text file.",
    )

    args = parser.parse_args()
    main(args)