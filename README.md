# SpIn-ViT: Interpretability of Vision Transformers using SAE

**SpIn-ViT** is an ongoing research project aimed at exploring the interpretability of Vision Transformers (ViTs) by leveraging Sparse Autoencoders. In this project, we project the ViT's patch-level features into a higher dimensional space to encourage monosemanticity, allowing better interpretability of individual neurons/features.

## Setup and Execution Order

The project is structured into different modules. Before getting started, please download the python package manager globally on your terminal using the link [uv](https://docs.astral.sh/uv/getting-started/installation/) as your guide.

Then run these commands below once you are in the SpIn-ViT repository:
```bash
uv venv --python 3.11
uv sync
```

`uv sync` installs every package from `uv.lock` into the virtual environment, so run it before any scripts (and again whenever `pyproject.toml` or `uv.lock` changes).

### Step 1: Prepare Data

Datasets are either pulled automatically via [HuggingFace Datasets](https://huggingface.co/docs/datasets) or loaded from a local folder, as configured in `src/utils/train.py::get_dataloader`:

| Dataset | Source |
|---|---|
| Flowers102 | HuggingFace: `pufanyi/flowers102` |
| Stanford Cars | **Local** (`data/raw/cars/`) |
| FGVC-Aircraft | **Local** (`data/raw/fgvc/`) |
| Caltech-101 | HuggingFace: `HuggingFaceM4/Caltech-101` |
| Food101 | HuggingFace: `ethz/food101` |
| SUN397 | HuggingFace: `tanganke/sun397` |
| DTD | HuggingFace: `tanganke/dtd` |
| EuroSAT | HuggingFace: `timm/eurosat-rgb` |
| OxfordPets | HuggingFace: `timm/oxford-iiit-pet` |

HuggingFace datasets need no manual download — they are fetched and cached on first run.

#### Local datasets

Download these and place them under `data/raw/` before training:

**Stanford Cars** — [https://github.com/cyizhuo/Stanford_Cars_dataset](https://github.com/cyizhuo/Stanford_Cars_dataset)

```
data/raw/cars/
├── train/<class name>/*.jpg
└── test/<class name>/*.jpg
```

**FGVC-Aircraft** — [https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) (sort the images into one folder per variant)

```
data/raw/fgvc/
├── train/<variant>/*.jpg
├── train.csv   # columns: filename,Classes,Labels
└── val.csv
```

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

### Step 3: SpIn-ViT Inference and Patch Figures

Once a model has been trained, run inference to identify, for every neuron in the sparse dictionary, the top-k images that activate it most strongly. The same command then saves a patch-highlight figure for each neuron/image pair:

```bash
# Get top-k images per neuron and save patch figures
uv run src/main.py \
    --train "false" \
    --dataset_name "flowers102" \
    --vit_sae_model_pth "model-vitsae/flowers102/best_topk_model.pth" \
    --activations_dir "activations/flowers102" \
    --output_dir "record-vitsae/flowers102" \
    --output_txt_file "record-vitsae/flowers102/top_k_neurons.txt" \
    --interp_dir "interpretability"
```

- Top-k neurons are written to `--output_txt_file` as `neuron:image_index` lines.
- Figures are saved to `interpretability/<dataset_name>/neuron_<idx>_<line>.png`, showing the original image next to the patches where that SpIn-ViT neuron fires (activation > 0.5) highlighted in yellow.
- Patch figures currently load images from the Flowers102 test split, so use them with Flowers102 models.

To save only the figures from an existing top-k file (no training or inference), add `--interpret`; `--max_images N` caps how many are saved:

```bash
uv run src/main.py --interpret --max_images 50
```

### Step 4: Run FOLD-SE-M to Create Rulesets

`src/foldse_main.py` exports binarized CLS-token SAE features for each dataset to CSV and sends them to FOLD-SE-M to learn rulesets. Set your FOLD-SE credentials first:

```bash
export FOLDSE_USERNAME="<registered email>"
export FOLDSE_PASSWORD="<password>"

uv run src/foldse_main.py --fetch           # features -> foldse/<dataset>_{train,test}.csv
uv run src/foldse_main.py --train           # rules    -> foldse_res/<dataset>/full_res.json
uv run src/foldse_main.py --fetch --train   # both
```

Train CSVs use ground-truth labels; test CSVs use the model's predicted labels, so the rules explain the model rather than the data.
