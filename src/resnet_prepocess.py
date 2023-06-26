import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import pandas as pd
import torch


BASE_PATH = "/home/nlp/ron.eliav/pose3d"

TRAIN_DATA_PATHES = [
    f"{BASE_PATH}/data/train_images-1",
    f"{BASE_PATH}/data/train_images-2",
    f"{BASE_PATH}/data/train_images-3",
    ]
TRAIN_LABELS_PATH = f"{BASE_PATH}/data/train_labels-1:3.csv"

DEV_DATA_PATH = f"{BASE_PATH}/data/train_images-4"


RESNET_PREPOCESS = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def get_images_from_dir(path):
    images = {}
    image_pathes = os.listdir(path)
    for image_path in image_pathes:
        im = Image.open(f"{path}/{image_path}")
        images[image_path] = RESNET_PREPOCESS(im)
        im.close()

    return images

def get_data(path):
    if isinstance(path, str):
        return get_images_from_dir(path)
    images = {}
    for p in path:
        images.update(get_data(p))
    return images


# def get_labels(path):
#     df = pd.read_csv(path)
#     return df
    # df[['Easting', 'Northing', 'Height', 'Roll', 'Pitch', 'Yaw']]


# def prepocess(prepocessimage):
    
#     return RESNET_PREPOCESS(image)


def get_pre_data(data_pathes, prepocess_approach='resnet'):
    images = get_data(data_pathes) # a dict
    # df = get_labels(labels_path) # a df
    # df['image'] = df['Filename'].map(images)
    # df = df.loc[df['image'].notna()]
    # prepocesses_data = {}
    # if prepocess_approach == 'resnet':
    #     for k, v in images.items():
    #         prepocesses_data[k] = RESNET_PREPOCESS(v)
    #         v.close()
        # df['prepocess_image'] = df['image'].apply(RESNET_PREPOCESS)
    return images

def save_prepocess_data(data_path, output_path):
    data = get_pre_data(data_path)
    for filename, image in data.items():
        torch.save(image, f"{output_path}/{filename}.pt")
    # dev_loader = DataLoader(CustomDataset(get_data_and_labels(dev_data, dev_labels)), batch_size=batch_size, shuffle=False)
    # return train_loader, dev_loader


save_prepocess_data(DEV_DATA_PATH, f"{BASE_PATH}/data/resnet_dev_data")