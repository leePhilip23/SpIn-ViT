# SpIn-ViT: Interpretability of Vision Transformers using SAE

**SpIn-ViT** is an ongoing research project aimed at exploring the interpretability of Vision Transformers (ViTs) by leveraging Sparse Autoencoders. In this project, we project the ViT's patch-level features into a higher dimensional space to encourage monosemanticity, allowing better interpretability of individual neurons/features.

## Setup and Execution Order

The project is structured into different modules. Before getting started, please download the python package manager globally on your terminal using the link [uv](https://docs.astral.sh/uv/getting-started/installation/) as your guide.

Then run these commands below once you are in the SpIn-ViT repository:
```bash
uv venv --python 3.11
uv lock
```

This will ensure that all the packages are downloaded in your virtual environment before running any scripts.

### Step 1: Prepare Data

Datasets are pulled automatically via [HuggingFace Datasets](https://huggingface.co/docs/datasets) (e.g. `timm/oxford-iiit-pet`) or loaded from a local folder structure, depending on which dataset is configured in `src/utils/train.py::get_dataloader`. Supported datasets include:

- Oxford-IIIT Pets
- Flowers102
- Stanford Cars
- FGVC-Aircraft
- Caltech-101
- Food101
- SUN397
- DTD
- EuroSAT
- GTSRB
- Places365
- Cats vs. Dogs

For datasets loaded from disk (e.g. Cars, FGVC, SUN397, Places365, GTSRB), place the raw data under `data/raw/<dataset_name>/` before training. For HuggingFace-backed datasets, no manual download is required — they will be fetched and cached automatically on first run.

### Step 2: Train the SpIn-ViT Model

```bash
# Train SpIn-ViT (dataset is selected in src/utils/train.py::get_dataloader)
uv run src/main.py \
    --train "true" \
    --dataset_name "pets" \
    --model_dir "model-vitsae" \
    --log_dir "record-vitsae" \
    --num_epochs 50
```

This trains a ViT backbone jointly with a sparse autoencoder head, saving the best checkpoint to `model-vitsae/<dataset_name>/` and writing training logs to `record-vitsae/log.txt`.

### Step 3: SpIn-ViT Inference

Once a model has been trained, run inference to identify, for every neuron in the sparse dictionary, the top-k images that activate it most strongly:

```bash
# Get top-k images per neuron for the trained SpIn-ViT model
uv run src/main.py \
    --train "false" \
    --dataset_name "pets" \
    --vit_sae_model_pth "model-vitsae/pets/best_model_pets.pth" \
    --activations_dir "activations/pets" \
    --output_dir "record-vitsae/pets" \
    --output_txt_file "record-vitsae/pets/top_k_neurons.txt"
```

### Step 4: Inspect Attention Maps and Run FOLD-SE-M to Create Rulesets

Use `src/spinvit-patches-foldse.ipynb` to visualize per-neuron attention/activation maps over the top-k images produced in Step 3, and to inspect qualitative examples of monosemantic vs. polysemantic neurons.
