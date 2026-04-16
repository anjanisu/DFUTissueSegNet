import sys
import os

# Local paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CODES_DIR = os.path.join(BASE_DIR, 'Codes')

# Add Codes to sys.path to use local segmentation_models_pytorch
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import cv2
import numpy as np
import segmentation_models_pytorch as smp
print(f"Using segmentation_models_pytorch from: {smp.__file__}")
from segmentation_models_pytorch.utils import metrics, losses, base
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from datetime import datetime
import torch.nn.functional as F
from PIL import Image

# Dataset directories
DATASET_ROOT = os.path.join(BASE_DIR, 'DFUTissue')
UNLABELED_DIR = os.path.join(DATASET_ROOT, 'Unlabeled')
OUTPUT_DIR_ROOT = os.path.join(BASE_DIR, 'Outputs')
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR_ROOT, 'checkpoints', 'MiT+pscse_mit_b3_sup')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR_ROOT, 'Predictions', 'Phase1')
PALETTE_DIR = os.path.join(OUTPUT_DIR_ROOT, 'Predictions_Palette', 'Phase1')

os.makedirs(PREDICTIONS_DIR, exist_ok=True)
os.makedirs(PALETTE_DIR, exist_ok=True)

class Dataset(BaseDataset):
    def __init__(self, list_IDs, images_dir, preprocessing=None, resize=(False, (256, 256)), n_classes=4):
        self.ids = list_IDs
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.preprocessing = preprocessing
        self.resize = resize
        self.n_classes = n_classes

    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.resize[0]:
            image = cv2.resize(image, self.resize[1], interpolation=cv2.INTER_NEAREST)
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        return image

    def __len__(self):
        return len(self.ids)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [albu.Lambda(image=preprocessing_fn), albu.Lambda(image=to_tensor)]
    return albu.Compose(_transform)

# Parameters
BASE_MODEL = 'MiT+pscse'
ENCODER = 'mit_b3'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 1 
n_classes = 4
ACTIVATION = 'sigmoid'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TO_CATEGORICAL = True

# Colors for palette: [Granulation(Red), Callus(Green), Fibrin(Blue), Background(Gray)]
# Reference palette from notebook: [[128, 128, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
palette = [[128, 128, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

# Load model
print(f"Using device: {DEVICE}")
model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=n_classes, activation=ACTIVATION, decoder_attention_type='pscse')
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
model.to(DEVICE)

checkpoint_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
if not os.path.exists(checkpoint_path):
    print(f"Error: Checkpoint not found at {checkpoint_path}")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Unlabeled data
list_IDs_unlabeled = os.listdir(UNLABELED_DIR)
print(f"No. of unlabeled images: {len(list_IDs_unlabeled)}")

unlabeled_dataset = Dataset(list_IDs_unlabeled, UNLABELED_DIR, preprocessing=get_preprocessing(preprocessing_fn))
unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Starting Pseudo-Label Generation...")
with torch.no_grad():
    for i, image in enumerate(unlabeled_dataloader):
        filename = list_IDs_unlabeled[i]
        print(f"Processing ({i+1}/{len(list_IDs_unlabeled)}): {filename}")
        
        pr_mask = model.predict(image.to(DEVICE))
        
        if TO_CATEGORICAL:
            pr_mask = torch.argmax(pr_mask, dim=1)
            
        pred = pr_mask.squeeze().cpu().numpy().astype(np.uint8)
        
        # Save raw prediction
        cv2.imwrite(os.path.join(PREDICTIONS_DIR, filename), pred)
        
        # Save palette prediction
        pal_pred = Image.fromarray(pred).convert("P")
        pal_pred.putpalette(np.array(palette, dtype=np.uint8))
        pal_pred.save(os.path.join(PALETTE_DIR, filename))

print(f"\nPseudo-labels saved to {PREDICTIONS_DIR}")
print(f"Palette labels saved to {PALETTE_DIR}")
