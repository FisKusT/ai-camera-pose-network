import sys
import random
import numpy as np
import torch
from torchvision.io import read_image
import pytorch_lightning as pl
from pytorch_lightning import Trainer

# sys.path.append("/home/nlp/ron.eliav/pose3d/ai-camera-pose-network/src/")
from model import PoseDetector
from utilities import get_data_loaders


BASE_PATH = "/home/nlp/ron.eliav/pose3d"

TRAIN_DATA_PATHES = [
    f"{BASE_PATH}/data/train_images-1",
    f"{BASE_PATH}/data/train_images-2",
    f"{BASE_PATH}/data/train_images-3",
    ]
TRAIN_LABELS_PATH = f"{BASE_PATH}/data/train_labels-1:3.csv"


DEV_DATA_PATH = f"{BASE_PATH}/data/train_images-4"
DEV_LABELS_PATH = f"{BASE_PATH}/data/train_labels-4.csv"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)




seed_everything(seed=42)
model = PoseDetector()
train_loader, dev_loader = get_data_loaders(TRAIN_DATA_PATHES, TRAIN_LABELS_PATH, DEV_DATA_PATH, DEV_LABELS_PATH)

print("start training")
trainer = Trainer(max_epochs=5, devices=[0,1,2,3], accelerator="gpu", num_sanity_val_steps=2, val_check_interval=0.2) # , callbacks=checkpoint_callbacks, accumulate_grad_batches=1) # , strategy="deepspeed_stage_2_offload", 
# continue from checkpoint
trainer.fit(model, train_loader, dev_loader)