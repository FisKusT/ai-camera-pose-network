import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch



class CustomDataset(Dataset):
    def __init__(self, df):
        self.images = [self._prepocess(im) for im in df['image'].tolist()]
        self.data = df.drop('image', axis=1)
        self.labels = df[['Easting', 'Northing', 'Height', 'Roll', 'Pitch', 'Yaw']].values

    def _prepocess(self, images):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return preprocess(images)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], torch.tensor(self.labels[idx])


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


def get_data_and_labels(data_pathes, labels_path):
    images = get_data(data_pathes) # a dict
    df = get_labels(labels_path) # a df
    df['image'] = df['Filename'].map(images)
    df = df.loc[df['image'].notna()]
    return df

def get_data_loaders(train_data_pathes, train_labels_path, dev_data_pathes, dev_labels_path):
    train_loader = DataLoader(CustomDataset(get_data_and_labels(train_data_pathes, train_labels_path)), batch_size=32, shuffle=True)
    dev_loader = DataLoader(CustomDataset(get_data_and_labels(dev_data_pathes, dev_labels_path)), batch_size=32, shuffle=False)
    return train_loader, dev_loader