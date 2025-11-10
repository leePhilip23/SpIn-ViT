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
from PIL import Image

random.seed(1)
torch.manual_seed(1)

class Places365_train_test(VisionDataset):
    """
    Manages loading and transforming the data for the Places365 dataset.
    Mimics the other VisionDataset classes available in torchvision.datasets.
    Classes used by Townsend et. al.: bathroom, bedroom, kitchen
    """

    base_folder = "places365"

    def __init__(self,
                 root: str,
                 class_list = [],
                 transform = None,
                 target_transform = None,
                 ) -> None:

        # Call superconstructor
        super(Places365_train_test, self).__init__(root, transform=transform, target_transform=target_transform)
        # the class list is sorted
        self.test_indices = []
        self.class_list = sorted(class_list)
        # print(self.class_list)
        self.class_to_idx_dict = {}
        for i in range(len(self.class_list)):
            self.class_to_idx_dict[self.class_list[i]] = i


        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        target = self.targets[index]

        # Conforming to other datasets
        image = PIL.Image.fromarray(self.data[index])

        # If a transform was provided, transform the image
        if need_PIL == False:
            image = self.transform(image)
        else:
            img_transform = T.Compose([T.Resize((224,224))])
            image = img_transform(image)
        # If a target transform was provided, transform the target
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading data")
        # Get path of extracted dataset
        image_dirs_path = os.path.join(\
            #"..", 
            "data", "raw", "places365", "data_256")
        #Get all the images
        images_dict = {}
        self.train_test_idx_dict = {}
        for c in self.class_list:
            images_dict[c] = []

        for c in self.class_list:
            first_letter = c[0]
            for alphabet_dir in os.listdir(image_dirs_path):
                if alphabet_dir == first_letter:
                    alphabet_dir_path = os.path.join(image_dirs_path, alphabet_dir)
                    for class_dir in os.listdir(alphabet_dir_path):
                        if class_dir == c:
                            class_dir_path = os.path.join(alphabet_dir_path, class_dir)
                            images_dict[c] = images_dict[c] + os.listdir(class_dir_path)
                            break
                    break
        # Convert the images to numpy arrays
        targets = []
        image_np_list = []
        for c, im_list in images_dict.items():
            for im_idx in range(len(im_list)):
                image_path = os.path.join(image_dirs_path, c[0], c, im_list[im_idx])
                image = PIL.Image.open(image_path)
                image_numpy = np.array(image)
                image_np_list.append(image_numpy)
                targets.append(self.class_to_idx_dict[c])
            self.train_test_idx_dict[c] = list(range(len(targets) - len(im_list), len(targets)))
            # self.test_indices = self.test_indices + list(range(len(image_np_list) - 1000, len(image_np_list)))
            # print(list(range(len(image_np_list) - 1000, len(image_np_list) - 1)))
        self.targets = targets
        self.data = image_np_list
        return image_np_list, targets
    
class Places365_val(VisionDataset):
    """
    Manages loading and transforming the data for the Places365 dataset.
    Mimics the other VisionDataset classes available in torchvision.datasets.
    Classes used by Townsend et. al.: Desert Road, Driveway, Forest Road, Highway, Street
    Corresponding class numbers are : {"desert_road" : 118, "driveway" : 127, "forest_road" : 152, "highway": 175, "street": 319}
    """

    base_folder = "places365"

    def __init__(self,
                 root: str,
                 class_list = [],
                 transform = None,
                 target_transform = None,
                 inference=False
                 ) -> None:

        # Call superconstructor
        super(Places365_val, self).__init__(root, transform=transform, target_transform=target_transform)

        self.class_num_dict = {"desert_road" : 118, "driveway" : 127, "forest_road" : 152, "highway": 175, "street": 319,
                               "bathroom" : 45,
                               "bedroom" : 52,
                               "kitchen" : 203,
                               "dining_room" : 121,
                               "living_room" : 215,
                               "office" : 244,
                               "home_office" : 176,
                               "waiting_room" : 352,
                               "conference_room" : 102,
                                "hotel_room" : 182,
                               "beach" : 48,
                               "desert" : 116,
                               "lake" : 205,
                               "tree_house" : 339,
                               "beach_house" : 49,
                               "forest" : 150
                               }
        # self.class_num_dict = {"bathroom" : 45,
        #                        "bedroom" : 52,
        #                        "kitchen" : 203}
        self.inference = inference
        self.image_dirs_path = os.path.join(
            #"..", 
            "data", "raw", "places365", "val_256")
        self.txt_file = os.path.join(
            #"..",
            "data", "raw", "places365", "places365_val.txt")
        self.class_num_list = []
        self.class_list = sorted(class_list)
        for c in self.class_list:
            self.class_num_list.append(self.class_num_dict[c])
        self.class_num_to_idx_dict = {}
        for i in range(len(self.class_num_list)):
            self.class_num_to_idx_dict[self.class_num_list[i]] = i


        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        target = self.targets[index]

        # Conforming to other datasets
        image = PIL.Image.fromarray(self.data[index])
        image = image.convert('RGB')
        # If a transform was provided, transform the image
        if need_PIL == False:
            image = self.transform(image)
        else:
            img_transform = T.Compose([T.Resize((224,224))])
            image = img_transform(image)

        # If a target transform was provided, transform the target
        if self.target_transform is not None:
            target = self.target_transform(target)

        return self.files[index], image, target if self.inference else image, target

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading val data")
        # Get path of extracted dataset
        image_dirs_path = self.image_dirs_path
        with open(self.txt_file, "r") as f:
            val_imgs_list = f.readlines()
        val_dict = {}
        for s in val_imgs_list:
            t_list = s.split(" ")
            val_dict[t_list[0]] = int(t_list[1])
        #Get all the images
        images_dict = {}
        for c in self.class_num_list:
            images_dict[c] = []

        for img in os.listdir(image_dirs_path):
            if val_dict[img] in self.class_num_list:
                images_dict[val_dict[img]].append(img)
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
        files = []
        targets = []
        image_np_list = []
        for c, im_list in images_dict.items():
            for im_idx in range(len(im_list)):
                image_path = os.path.join(image_dirs_path, im_list[im_idx])
                files.append(image_path)
                image = PIL.Image.open(image_path)
                image_numpy = np.array(image)
                image_np_list.append(image_numpy)
                targets.append(self.class_num_to_idx_dict[c])
        self.targets = targets
        self.data = image_np_list
        self.files = files
        return files, image_np_list, targets

class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """



    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
        self.indices = np.array(indices)

    def __getitem__(self, idx, need_PIL=False):

        image = self.dataset[idx][0]
        target = self.targets[idx]
        return image, target

    def __len__(self):
        return len(self.targets)


def PLACES_dataloaders(batch_size = 8, train_shuffle = True, class_list = None, inference=False):
    # train_test_set = Places365_train_test(root= "./", class_list=class_list, transform=T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))
    train_test_set = Places365_train_test(root=None, class_list=class_list, transform=T.Compose(
        [T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
         T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))
    # val_set = Places365_val(root= "./", class_list=class_list, transform=T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))

    val_set = Places365_val(root=None, class_list=class_list, transform=T.Compose(
        [T.Resize((224, 224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
         T.RandomHorizontalFlip(), T.RandomVerticalFlip()]), inference=inference)

    train_test_idx_dict =  train_test_set.train_test_idx_dict
    # create a train_idx_list and val_idx_list
    train_idx_dict = {}
    test_idx_dict = {}
    train_idx_list = []
    test_idx_list = []
    for c, idx_list in train_test_idx_dict.items():
        total_len = len(idx_list)
        # create a copy of idx_list and shuffle it
        idx_list_copy = idx_list.copy()
        random.shuffle(idx_list_copy)
        # split the shuffled list into two parts
        train_idx_dict[c] = idx_list_copy[:-1000]
        train_idx_list += idx_list_copy[:-1000]
        test_idx_dict[c] = idx_list_copy[-1000:]
        test_idx_list += idx_list_copy[-1000:]
    print(f"length of fullset: {len(train_test_set)}")
    # creating train, val and test set

    train_set = Subset(train_test_set, train_idx_list)
    test_set = Subset(train_test_set, test_idx_list)
    # train_set, test_set = create_train_test(train_test_set)
    print(f"train indices: {train_idx_list[:10]}")

    # print(f"val indices: {val_set.indices}")
    print(f"test indices: {test_idx_list[:10]}")

    train_loader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=train_shuffle, num_workers=4)

    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    test_loader = DataLoader(test_set, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    class_weights = [1] * len(class_list)
    return train_loader, val_loader, test_loader, len(class_list), torch.FloatTensor(class_weights)


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
            "..", 
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
        "..", 
        "data", 
        "raw"
    ),
                                 transform=T.Compose([T.ToTensor(), T.Resize((224, 224))]))

    test_set = GTSRB_test(os.path.join(
        "..", 
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


class ADE20k(VisionDataset):
    base_folder = "ADE20K_2021_17_01"

    def __init__(self,
                 root: str,
                 class_list = [],
                 transform = None,
                 target_transform = None,
                 ) -> None:

        # Call superconstructor
        super(ADE20k, self).__init__(root, transform=transform, target_transform=target_transform)

        # self.test_indices = []
        self.class_list = sorted(class_list)
        self.class_to_idx_dict = {}
        for i in range(len(self.class_list)):
            self.class_to_idx_dict[self.class_list[i]] = i


        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        target = self.targets[index]

        # Conforming to other datasets
        image = PIL.Image.fromarray(self.data[index][0])

        # If a transform was provided, transform the image
        if need_PIL == False:
            image = self.transform(image)
        else:
            img_transform = T.Compose([T.Resize((224,224))])
            image = img_transform(image)
        # If a target transform was provided, transform the target
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, self.data[index][1]), target

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading data")
        # Get path of extracted dataset
        image_dirs_path = os.path.join("ADE20K_2021_17_01", "images", "ADE", "training")
        #Get all the images
        images_dict = {}
        json_dict = {}
        self.train_test_idx_dict = {}
        for c in self.class_list:
            images_dict[c] = []

        # for c in self.class_list:
        for s_type in os.listdir(image_dirs_path):
            scene_path = os.path.join(image_dirs_path, s_type)
            for scene in os.listdir(scene_path):
                if scene in self.class_list:
                    scene_path = os.path.join(image_dirs_path, s_type, scene)
                    for image in os.listdir(scene_path):
                        if image.endswith(".jpg"):
                            images_dict[scene].append((os.path.join(scene_path, image), os.path.join(scene_path, image[:-4] + ".json")))

        # Convert the images to numpy arrays
        # self.idx_image_json_dict = {}
        image_np_list = []
        targets = []
        for c, im_list in images_dict.items():
            # self.scene_image_json_dict[c] = []
            for im_idx in range(len(im_list)):
                image_path = im_list[im_idx][0]
                image = PIL.Image.open(image_path)
                image = image.convert("RGB")
                image_np = np.array(image)

                # with open(im_list[im_idx][1], encoding="utf-8") as f:
                #     info_dict = json.load(f)
                # self.scene_image_json_dict[c].append((image, info_dict))
                image_np_list.append((image_np, im_list[im_idx][1]))
                targets.append(self.class_to_idx_dict[c])
            # self.train_test_idx_dict[c] = list(range(len(targets) - len(im_list), len(targets)))
            # self.test_indices = self.test_indices + list(range(len(image_np_list) - 1000, len(image_np_list)))
            # print(list(range(len(image_np_list) - 1000, len(image_np_list) - 1)))
        self.targets = targets
        self.data = image_np_list
        return image_np_list, targets


from transformers import ViTImageProcessorFast, CLIPImageProcessorFast
from torchvision.transforms import functional as F
from torchvision.transforms import RandomResizedCrop
import random

class CatsDogsDataset(Dataset):
    def __init__(self, file_list, transform=None, inference=False):
        print("Inference", inference)
        self.feat_extract = ViTImageProcessorFast.from_pretrained('google/vit-base-patch16-224')
        self.inference = inference
        if transform == 'train':
            self.transforms = T.Compose(
                [
                    T.Resize((224, 224)),
                    #T.RandomResizedCrop(224),
                    #T.RandomHorizontalFlip(),
                    T.ToTensor(),
                ]
            )
        else:
            self.transforms = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                ]
            )

        self.file_list = file_list

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            #img_transformed = self.transforms(img)
            #img_transformed = img_transformed / 255.0
            img_transformed = self.feat_extract(images=img, return_tensors='pt').pixel_values[0]

        label = img_path.split("/")[-2].split(".")[0]
        #label = float(1) if label == "dog" else float(0)
        label = 1 if label == "Dog" else 0

        return (img_path, img_transformed, torch.as_tensor(label)) if self.inference else (img_transformed, torch.as_tensor(label))
        #return (str(idx), img_transformed, float(self.labels[idx])) if self.inference else (img_transformed, torch.as_tensor(float(self.labels[idx])))


class Flowers102Dataset(Dataset):
    def __init__(self, split, inference=False):
        self.split = split
        self.inference = inference
        self.process =  ViTImageProcessorFast.from_pretrained(
            "google/vit-base-patch16-224", use_fast=True
        )
        self.data = self._load_data(split)
        
    def _load_data(self, split):
        abs_path = os.path.join("data", "raw", "flowers102", split)
        abs_dir = os.listdir(abs_path)

        img_label = []
        for label in abs_dir:
            img_dir = os.path.join(abs_path, label)
            img_paths = os.listdir(img_dir)
            for img in img_paths:
                img_label.append((os.path.join(img_dir, img), int(label)-1))
        
        return img_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path)
        image = self.process(images=image, return_tensors='pt').pixel_values[0]
        return (img_path, image, torch.as_tensor(label)) if self.inference else (image, torch.as_tensor(label))
    
import pickle

class HuggingFaceData(Dataset):
    def __init__(self, hf_data, split='train', inference=False, too_big=False):
        self.split = split
        self.inference = inference
        self.too_big = too_big
        self.hf_data = hf_data
        self.imgs_labels = self._load_data(hf_data, split)
        #self.process =  ViTImageProcessorFast.from_pretrained(
        #    "google/vit-base-patch16-224", use_fast=True
        #)
        self.process = CLIPImageProcessorFast.from_pretrained(
            "google/vit-base-patch16-224", use_fast=True
        )

    def _load_data(self, hf_data, split):
        imgs_labels = []
        if self.too_big and os.path.exists(f"data/raw/dtd/{split}.npy"):
            imgs_labels = np.load(f"data/raw/dtd/{split}.npy").tolist()
        else:
            for i in range(len(hf_data[split])):
                if self.too_big:
                    try:
                        img, label = i, hf_data[split][i]['label']
                    except:
                        print(f"Index {i} is corrupted")
                        continue
                    print(i)
                else:
                    try:
                        img, label = hf_data[split][i]['image'], hf_data[split][i]['label']
                    except:
                        print(f"Index {i} is corrupted")
                        continue
                    #print(i)
                imgs_labels.append((img, label))

        #with open("data/raw/food101.pkl", "wb") as f:
        #    pickle.dump(imgs_labels, f)

        #with open("data/raw/sun397.pkl", "rb") as f:
        #    imgs_labels = pickle.load(f)

            if self.too_big:
                arr = np.array(imgs_labels)
                np.save(f"data/raw/dtd/{split}.npy", arr)

        return imgs_labels

    def __len__(self):
        return len(self.imgs_labels)
    
    def __getitem__(self, idx):
       img, label = self.imgs_labels[idx]
       if self.too_big:
           img = self.hf_data[self.split][idx]['image']

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