import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import ViTImageProcessorFast


class HuggingFaceData(Dataset):
    def __init__(self, hf_data, split='train', inference=False, too_big=False):
        self.split = split
        self.inference = inference
        self.too_big = too_big
        self.process =  ViTImageProcessorFast.from_pretrained(
            "google/vit-base-patch16-224", use_fast=True
        )
        self.hf_data = hf_data
        self.imgs_labels = self._load_data(hf_data, split)


    def _load_data(self, hf_data, split):
        imgs_labels = []
        for i in range(len(hf_data[split])):
            try:
                img, label = i, hf_data[split][i]['label']
                _ = self.process(images=hf_data[split][i]['image'], return_tensors='pt').pixel_values[0]
            except:
                print(f"Image Index {i} is corrupted")
                continue
            imgs_labels.append((img, label))
        return imgs_labels

    def __len__(self):
        return len(self.imgs_labels)
    
    def __getitem__(self, idx):
        id, label = self.imgs_labels[idx]
        if self.too_big:
           img = self.hf_data[self.split][id]['image']
        else:
           img = id

        img = self.process(images=img, return_tensors='pt').pixel_values[0]
        label = torch.as_tensor(int(label) if not isinstance(label, int) else label)
        return (str(idx), img, label) if self.inference else (img, label)
        

class FolderData(Dataset):
    def __init__(self, dir_path, inference=False, csv=None):
        self.split = dir_path
        self.inference = inference
        self.csv = csv
        self.process = ViTImageProcessorFast.from_pretrained(
           "google/vit-base-patch16-224", use_fast=True
        )
        self.data = self._load_data(dir_path)

    def _load_data(self, abs_path):
        abs_folder = os.listdir(abs_path)

        if self.csv:
            df = pd.read_csv(self.csv)
            s = frozenset(df.filename.tolist())

        class_dict = {}
        for i, path in enumerate(sorted(abs_folder)):
            class_dict[path] = i

        img_class = []
        for class_path in abs_folder:
            imgs_directory = os.path.join(abs_path, class_path)
            imgs_folder = os.listdir(imgs_directory)

            for img in imgs_folder:
                if not self.csv:
                    img_class.append((os.path.join(imgs_directory, img), class_dict[class_path]))
                if self.csv and img in s:
                    img_class.append((os.path.join(imgs_directory, img), class_dict[class_path]))

        return img_class
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_file, label = self.data[idx]
        with Image.open(img_file) as img:
            img = img.convert('RGB')
        img = self.process(images=img, return_tensors='pt').pixel_values[0]
        label = torch.as_tensor(int(label) if not isinstance(label, int) else label)
        return (img_file, img, label) if self.inference else (img, label)