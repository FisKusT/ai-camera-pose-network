import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch



class CustomDataset(Dataset):
    def __init__(self, df):
        self.images = df['image'].tolist() # [self._prepocess(im) for im in df['image'].tolist()]
        self.data = df.drop('image', axis=1)
        self.labels = df[['Easting', 'Northing', 'Height', 'Roll', 'Pitch', 'Yaw']].values

    # def _resnet_prepocess(self, images):
    #     preprocess = transforms.Compose([
    #         transforms.Resize(256),
    #         transforms.CenterCrop(224),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     return preprocess(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx].float(), torch.tensor(self.labels[idx]).float()


def get_images_from_dir(path):
    images = {}
    image_pathes = os.listdir(path)
    for image_path in image_pathes:
        images[image_path] = Image.open(f"{path}/{image_path}")
    return images

def get_data(path):
    if isinstance(path, str):
        return get_images_from_dir(path)
    images = {}
    for p in path:
        images.update(get_data(p))
    return images

def get_labels(path):
    df = pd.read_csv(path)
    return df
    # df[['Easting', 'Northing', 'Height', 'Roll', 'Pitch', 'Yaw']]


def get_data_and_labels(data_path, labels_path, prepocess=False):
    if prepocess:
        images = get_data(data_path) # a dict
    else:
        images = {path.split(".pt")[0]: torch.load(f"{data_path}/{path}") for path in os.listdir(data_path)}
    df = get_labels(labels_path) # a df
    df['image'] = df['Filename'].map(images)
    df = df.loc[df['image'].notna()]
    return df


def get_data_loaders(train_data, train_labels, dev_data, dev_labels, batch_size):
    train_loader = DataLoader(CustomDataset(get_data_and_labels(train_data, train_labels)), batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(CustomDataset(get_data_and_labels(dev_data, dev_labels)), batch_size=batch_size, shuffle=False)
    return train_loader, dev_loader