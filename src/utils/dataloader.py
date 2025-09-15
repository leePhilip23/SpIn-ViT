import random
import pandas as pd
import PIL
import torch
import idx2numpy
from sklearn.model_selection import train_test_split
from torchvision.datasets import VisionDataset, VOCDetection
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from imageio.v2 import imread
import json
from torchvision.datasets import MNIST, GTSRB
import numpy as np
import os
import torchvision.transforms as T
import torchvision
random.seed(1)
torch.manual_seed(1)

class GTSRB_train_val(VisionDataset):

    """
    Manages loading and transforming the data for the GTSRB dataset.
    80:20 split for each class
    """
    base_folder = "GTSRB_train"

    def __init__(self,
                 root: str,
                 transform = None, 
                 num_classes=43,
                 samples_per_class=5
                 ) -> None:

        # Call superconstructor
        super(GTSRB_train_val, self).__init__(root, transform=transform)

        # self.val_indices = []
        # self.class_list = sorted(class_list)
        # self.class_to_idx_dict = {}
        # for i in range(len(self.class_list)):
        #     self.class_to_idx_dict[self.class_list[i]] = i

        self.samples_per_class = samples_per_class
        self.total_d_dim = self.samples_per_class * num_classes
        self.semgments = self.generate_segments()
        self.transform = transform
        # Load data with the given classes
        self.load_data()

    
    def generate_segments(self):
        if self.total_d_dim % self.samples_per_class != 0:
            raise ValueError("m must be divisible by n.")
        
        indices = list(range(self.total_d_dim))
        random.shuffle(indices)

        segments = [indices[i:i + self.samples_per_class] for i in range(0, self.total_d_dim, self.samples_per_class)]
        np.save(os.path.join(
            #"..", 
            "model-vitsae", 
            "GTSRB", 
            "segments.npy"
        ), segments)
        return segments
    

    def __getitem__(self, index, need_PIL=False):



        # Conforming to other datasets
        # image = PIL.Image.fromarray(self.data[index])
        #
        if type(index) is not list:
            target = self.targets[index]
            image = self.data[index]
            segment = self.semgments[target]
            binary_tensor = torch.zeros(self.total_d_dim)
            binary_tensor[segment] = 1
            # print(image.shape)
            # If a transform was provided, transform the image
            if need_PIL == False:
                image = self.transform(image)
                # print(image.shape)
            else:
                img_transform = T.Compose([T.Resize((224, 224))])
                image = img_transform(image)
            return image, binary_tensor, torch.tensor(segment), target
        else:
            image_list = []
            bin_tensor_list = []
            targets = []
            for i in index:
                image = self.data[i]
                target = self.targets[i]
                segment = self.semgments[target]
                binary_tensor = torch.zeros(self.total_d_dim)
                binary_tensor[segment] = 1
                # print(image.shape)
                # If a transform was provided, transform the image
                if need_PIL == False:
                    image = self.transform(image)
                    # print(image.shape)
                else:
                    img_transform = T.Compose([T.Resize((224, 224))])
                    image = img_transform(image)
                image_list.append(image)
                bin_tensor_list.append(segment)
                targets.append(target)
        return image_list, binary_tensor, torch.tensor(segment), targets

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading train and val data")
        # Get path of extracted dataset
        image_dirs_path = os.path.join(self.root, self.base_folder, "Final_Training", "Images")
        #Get all the images for each class
        images_dict = {}
        self.class_to_idx_dict = {"00000" : 0, "00001" : 1, "00002": 2,
                                  "00003": 3, "00004": 4, "00005": 5,
                                  "00006": 6, "00007": 7, "00008": 8,
                                  "00009": 9, "00010": 10, "00011": 11,
                                  "00012": 12, "00013": 13, "00014": 14,
                                  "00015": 15, "00016": 16, "00017": 17,
                                  "00018": 18, "00019": 19, "00020": 20,
                                  "00021": 21, "00022": 22, "00023": 23,
                                  "00024": 24, "00025": 25, "00026": 26,
                                  "00027": 27, "00028": 28, "00029": 29,
                                  "00030": 30, "00031": 31, "00032": 32,
                                  "00033": 33, "00034": 34, "00035": 35,
                                  "00036": 36, "00037": 37, "00038": 38,
                                  "00039": 39, "00040": 40, "00041": 41,
                                  "00042": 42}

        for i, c in enumerate(os.listdir(image_dirs_path)):
            images_dict[c] = os.listdir(os.path.join(image_dirs_path, c))
            # self.class_to_idx_dict[c] = i
        # for c in self.class_list:
        #     first_letter = c[0]
        #     for alphabet_dir in os.listdir(image_dirs_path):
        #         if alphabet_dir == first_letter:
        #             alphabet_dir_path = os.path.join(image_dirs_path, alphabet_dir)
        #             for class_dir in os.listdir(alphabet_dir_path):
        #                 if class_dir == c:
        #                     class_dir_path = os.path.join(alphabet_dir_path, class_dir)
        #                     images_dict[c] = images_dict[c] + os.listdir(class_dir_path)
        #                     break
        #             break
        # Convert the images to numpy arrays
        targets = []
        image_np_list = []
        self.class_indices_dict = {}
        for i, (c, im_list) in enumerate(images_dict.items()):
            for im_idx in range(len(im_list)):
                image_path = os.path.join(image_dirs_path, c, im_list[im_idx])
                if ".csv" in image_path:
                    continue
                image = imread(image_path)
                image_numpy = np.array(image)
                # image_numpy = np.moveaxis(image_numpy, -1, 0)
                # print(image_numpy.shape)
                image_np_list.append(image_numpy)
                targets.append(self.class_to_idx_dict[c])
            self.class_indices_dict[c] = list(range(len(targets) - len(im_list), len(targets)))
            # self.test_indices = self.test_indices + list(range(len(image_np_list) - 1000, len(image_np_list)))
            # print(list(range(len(image_np_list) - 1000, len(image_np_list) - 1)))
        self.targets = np.array(targets)
        self.data = image_np_list

class GTSRB_test(VisionDataset):


    base_folder = "GTSRB_test"

    def __init__(self,
                 root: str,
                 transform = None,
                 ) -> None:

        # Call superconstructor
        super(GTSRB_test, self).__init__(root, transform=transform)

        # self.class_num_dict = {"desert_road" : 118, "driveway" : 127, "forest_road" : 152, "highway": 175, "street": 319,
        #                        "bathroom" : 45,
        #                        "bedroom" : 52,
        #                        "kitchen" : 203}
        # self.class_num_dict = {"bathroom" : 45,
        #                        "bedroom" : 52,
        #                        "kitchen" : 203}
        # self.class_num_list = []
        # self.class_list = sorted(class_list)
        # for c in self.class_list:
        #     self.class_num_list.append(self.class_num_dict[c])
        # self.class_num_to_idx_dict = {}
        # for i in range(len(self.class_num_list)):
        #     self.class_num_to_idx_dict[self.class_num_list[i]] = i


        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        if type(index) is not list:
            target = self.targets[index]
            file = self.files[index]

            # Conforming to other datasets
            image = self.data[index]
            # If a transform was provided, transform the image
            if need_PIL == False:
                image = self.transform(image)
            else:
                img_transform = T.Compose([T.Resize((224,224))])
                image = img_transform(image)

            return file, image, target
        else:
            image_list = []
            targets = []
            for i in index:
                file = self.files[i]
                image = self.data[i]
                target = self.targets[i]
                # If a transform was provided, transform the image
                if need_PIL == False:
                    image = self.transform(image)
                else:
                    img_transform = T.Compose([T.Resize((224,224))])
                    image = img_transform(image)
                image_list.append(image)
                targets.append(target)
            return file, image_list, targets

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading test data")
        # Get path of extracted dataset
        image_dirs_path = os.path.join(self.root, self.base_folder, "Final_Test", "Images")
        df = pd.read_csv(os.path.join(self.root, self.base_folder, "GT-final_test.csv"), sep = ";")

        image_list = os.listdir(image_dirs_path)
        targets = []
        for img in image_list:
            # print(img)
            if ".csv" in img:
                continue
            targets.append(df["ClassId"][df["Filename"] == img].values[0])

        # for c in self.class_list:
        #     first_letter = c[0]
        #     for alphabet_dir in os.listdir(image_dirs_path):
        #         if alphabet_dir == first_letter:
        #             alphabet_dir_path = os.path.join(image_dirs_path, alphabet_dir)
        #             for class_dir in os.listdir(alphabet_dir_path):
        #                 if class_dir == c:
        #                     class_dir_path = os.path.join(alphabet_dir_path, class_dir)
        #                     images_dict[c] = images_dict[c] + os.listdir(class_dir_path)
        #                     break
        #             break

        image_np_list, files = [], []
        for im in image_list:
            if ".csv" in im:
                continue

            image_path = os.path.join(image_dirs_path, im)
            files.append(image_path[:-4])
            image = imread(image_path)
            image_numpy = np.array(image)
            # image_numpy = np.moveaxis(image_numpy, -1, 0)
            image_np_list.append(image_numpy)
            # targets.append(self.class_num_to_idx_dict[c])
        self.targets = np.array(targets)
        # print(image_np_list)
        self.files = files
        self.data = image_np_list

def GTSRB_dataloaders(batch_size = 8, train_shuffle = True):
    train_val_set = GTSRB_train_val(os.path.join(
        #"..", 
        "data", 
        "raw"
    ),
                                 transform=T.Compose([T.ToTensor(), T.Resize((224, 224))]))

    test_set = GTSRB_test(os.path.join(
        #"..", 
        "data", 
        "raw"
    ),
                           transform=T.Compose([T.ToTensor(), T.Resize((224, 224))]))
    train_val_idx_dict = train_val_set.class_indices_dict
    #create a train_idx_list and val_idx_list
    train_idx_dict = {}
    val_idx_dict = {}
    train_idx_list = []
    val_idx_list = []
    for c, idx_list in train_val_idx_dict.items():
        total_len = len(idx_list)
        # create a copy of idx_list and shuffle it
        idx_list_copy = idx_list.copy()
        random.shuffle(idx_list_copy)
        # split the shuffled list into two parts
        train_idx_dict[c] = idx_list_copy[:int(0.8 * total_len)]
        train_idx_list += idx_list_copy[:int(0.8 * total_len)]
        val_idx_dict[c] = idx_list_copy[int(0.8 * total_len):]
        val_idx_list += idx_list_copy[int(0.8 * total_len):]

    #create a train_data and val_data subsets from the train_val_data and train_idx_list and val_idx_list
    train_set = Subset(train_val_set, train_idx_list)
    val_set = Subset(train_val_set, val_idx_list)
    print("train indices", train_idx_list[:10])
    print("val indices", val_idx_list[:10])
    # create train_loader and val_loader and test_loader
    pin_memory = True if torch.cuda.is_available() else False

    # TODO: Change batch size to 8 for training supervised contrastive loss
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=train_shuffle, num_workers=4, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=pin_memory)

    #calculate weights for each class
    class_weights = []
    nt = len(train_idx_list)
    for c, idx_list in train_idx_dict.items():
        nc = len(idx_list)
        class_weights.append(1 - (nc/nt))
    return train_loader, val_loader, test_loader, 43, torch.FloatTensor(class_weights)


import os
import scipy.io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTImageProcessor


class Flowers102Dataset(Dataset):
    def __init__(self, split="train", transform=None):
        self.split = split
        self.transform = transform
        self.process =  ViTImageProcessor.from_pretrained(
            "google/vit-base-patch16-224", use_fast=True
        )

        # Load metadata
        labels_path = os.path.join("data", "flowers-102", "imagelabels.mat")
        setid_path = os.path.join("data", "flowers-102", "setid.mat")

        # Load .mat files
        self.labels = scipy.io.loadmat(labels_path)["labels"][0]  # 1-indexed
        setid = scipy.io.loadmat(setid_path)

        # Get indices for split
        if split == "train":
            self.indices = setid["trnid"][0] - 1  # Convert to 0-indexed
        elif split == "val":
            self.indices = setid["valid"][0] - 1
        elif split == "test":
            self.indices = setid["tstid"][0] - 1
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        img_path = os.path.join("data", "flowers-102", "jpg", f"image_{actual_idx + 1:05d}.jpg")
        label = self.labels[actual_idx] - 1  # Convert to 0-indexed

        image = Image.open(img_path)
        image = self.process(image).pixel_values
        
        if self.transform:
            image = self.transform(image)

        if self.split == "test":
            return f"image_{actual_idx + 1:05d}", torch.as_tensor(image[0]), torch.as_tensor(label)
        return torch.as_tensor(image[0]), torch.as_tensor(label)