import os
import copy
import argparse
import logging
import torch
from utils.train import get_dataloader, train_model
from utils.sae_inference import get_top_k


def main(args):
    if args.train == "true":
        # Create directories for model and log storage
        print("Training the model...")
        datasets = [
            "flowers102",
            #"cars",
            #"fgvc",
            "caltech101",
            "foods101",
            "suns397",
            "dtd",
            "eurosat",
            "pets"
        ]

        if args.dataset_name == "any":
            for i, d in enumerate(datasets):
                curr_args = copy.deepcopy(args)
                curr_args.dataset_name = d

                os.makedirs(args.model_dir, exist_ok=True)
                os.makedirs(os.path.join(args.model_dir, args.dataset_name), exist_ok=True)
                os.makedirs(args.log_dir, exist_ok=True)

                logging_path = os.path.join(args.log_dir,"log.txt")
                logging.basicConfig(filename=logging_path, level=logging.INFO, format='%(message)s')

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                train_dataloader, val_dataloader, _, num_classes, _ = get_dataloader(False, iteration=i)
                train_model(curr_args, device, train_dataloader, val_dataloader, num_classes)

            get_top_k(args)
        else:
            i = datasets.index(args.dataset_name)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_dataloader, val_dataloader, _, num_classes, _ = get_dataloader(False, iteration=i)
            train_model(args, device, train_dataloader, val_dataloader, num_classes)

    else:
        print("Getting top-k neurons...")
        get_top_k(args)
    

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a ViT model")
    parser.add_argument("--train", type=str, default="true", help="Whether to train the model or not")

    # Train parameters
    parser.add_argument("--dataset_name", type=str, default="any", help="The dataset used")
    parser.add_argument("--model_dir", type=str, default="model-vitsae", help="The directory where the model will be saved.")
    parser.add_argument("--log_dir", type=str, default="record-vitsae", help="The directory where the log will be saved.")
    parser.add_argument("--num_epochs", type=int, default=50, help="The number of epochs to train the model.")

    # Inference parameters
    parser.add_argument(
        "--vit_sae_model_pth",
        type=str,
        default=os.path.join("model-vitsae", "flowers102", "best_topk_model.pth"),
        help="Path to the ViViT model file.",
    )
    parser.add_argument(
        "--activations_dir",
        type=str,
        default=os.path.join("activations", "flowers102"),
        help="Directory to save the activations files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join("record-vitsae", "flowers102"),
        help="Directory to save the output files.",
    )
    parser.add_argument(
        "--output_txt_file",
        type=str,
        default=os.path.join("record-vitsae", "flowers102", "top_k_neurons.txt"),
        help="Path to the output text file.",
    )

    args = parser.parse_args()
    main(args)