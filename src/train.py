import sys
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# sys.path.append("/home/nlp/ron.eliav/pose3d/ai-camera-pose-network/src/")
from model import PoseDetector
from utilities import get_data_loaders


BATCH_SIZE = 64


BASE_PATH = "/home/nlp/ron.eliav/pose3d"

# TRAIN_DATA_PATHES = [
#     f"{BASE_PATH}/data/train_images-1",
#     f"{BASE_PATH}/data/train_images-2",
#     f"{BASE_PATH}/data/train_images-3",
#     ]
TRAIN_DATA_PATHES = BASE_PATH + "/data/resnet_data"

TRAIN_LABELS_PATH = f"{BASE_PATH}/data/train_labels-1:3.csv"


DEV_DATA_PATH = BASE_PATH + "/data/resnet_dev_data" # f"{BASE_PATH}/data/train_images-4"
DEV_LABELS_PATH = f"{BASE_PATH}/data/train_labels-4.csv"



config = {
    'train_data': TRAIN_DATA_PATHES,
    'train_labels': TRAIN_LABELS_PATH,
    'dev_data': DEV_DATA_PATH,
    'dev_labels': DEV_LABELS_PATH,
    'batch_size': BATCH_SIZE,
}



def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)



checkpoint_callback = ModelCheckpoint(
    # dirpath='checkpoints/',
    filename='{epoch}-{step}',
    monitor='val_loss',
    mode='min',
    save_top_k=20  # Save all checkpoints
)


seed_everything(seed=42)
model = PoseDetector()
train_loader, dev_loader = get_data_loaders(**config)

print("start training")
trainer = Trainer(max_epochs=500, devices=[0,1,2,3], accelerator="gpu", num_sanity_val_steps=2, callbacks=[checkpoint_callback]) # , callbacks=checkpoint_callbacks, accumulate_grad_batches=1) # , strategy="deepspeed_stage_2_offload", 
trainer.fit(model, train_loader, dev_loader)