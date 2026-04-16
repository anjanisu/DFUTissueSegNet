import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CODES_DIR = os.path.join(BASE_DIR, 'Codes')
if CODES_DIR not in sys.path: sys.path.insert(0, CODES_DIR)

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import pandas as pd
from PIL import Image

DATASET_ROOT = os.path.join(BASE_DIR, 'DFUTissue')
UNLABELED_DIR = os.path.join(DATASET_ROOT, 'Unlabeled')
OUTPUT_DIR_ROOT = os.path.join(BASE_DIR, 'Outputs')

# Discover best Phase 1 run from output.txt or Results_Phase3
RESULTS_PHASE3_DIR = os.path.join(OUTPUT_DIR_ROOT, 'Results_Phase3')
CHECKPOINT_PHASE3_DIR = os.path.join(OUTPUT_DIR_ROOT, 'checkpoints_Phase3')

# Parse best run from output.txt
with open(os.path.join(BASE_DIR, 'output.txt'), 'r') as f:
    best_run = None
    for line in f:
        if "Best Run:" in line and "validation loss:" in line:
            best_run = line.split("Best Run:")[1].split("with")[0].strip()

best_run_idx = int(best_run)
best_run_checkpoint = os.path.join(CHECKPOINT_PHASE3_DIR, f'MiT+pscse_padded_mit_b3_run_{best_run_idx}_selfSupervised', 'best_model.pth')
phase1_unsup_txt = os.path.join(RESULTS_PHASE3_DIR, f'run_{best_run_idx}_unsup_train.txt')

with open(phase1_unsup_txt, 'r') as f:
    phase1_names = [line.strip() for line in f if line.strip()]

all_unlabeled = os.listdir(UNLABELED_DIR)
phase2_unlabeled = [x for x in all_unlabeled if x not in phase1_names]

PREDICTIONS_PHASE2 = os.path.join(OUTPUT_DIR_ROOT, 'Predictions', 'Phase2')
PALETTE_PHASE2 = os.path.join(OUTPUT_DIR_ROOT, 'Predictions_Palette', 'Phase2')
os.makedirs(PREDICTIONS_PHASE2, exist_ok=True)
os.makedirs(PALETTE_PHASE2, exist_ok=True)

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
        if self.resize[0]: image = cv2.resize(image, self.resize[1], interpolation=cv2.INTER_NEAREST)
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']
        return image

    def __len__(self): return len(self.ids)

def to_tensor(x, **kwargs): return x.transpose(2, 0, 1).astype('float32')
def get_preprocessing(preprocessing_fn):
    return albu.Compose([albu.Lambda(image=preprocessing_fn), albu.Lambda(image=to_tensor)])

BASE_MODEL = 'MiT+pscse'
ENCODER = 'mit_b3'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 4
n_classes = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=n_classes, activation='sigmoid', decoder_attention_type='pscse')
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
model.to(DEVICE)
model.load_state_dict(torch.load(best_run_checkpoint, map_location=DEVICE)['state_dict'])
model.eval()

dataset = Dataset(phase2_unlabeled, UNLABELED_DIR, preprocessing=get_preprocessing(preprocessing_fn))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

palette = [[128, 128, 128], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

print(f"Generating Phase 2 Pseudo-labels for {len(phase2_unlabeled)} images using Best Phase 1 Model (Run {best_run_idx})...")
with torch.no_grad():
    for i, image in enumerate(dataloader):
        filename = phase2_unlabeled[i]
        pr_mask = model.predict(image.to(DEVICE))
        pr_mask = torch.argmax(pr_mask, dim=1)
        pred = pr_mask.squeeze().cpu().numpy().astype(np.uint8)
        
        cv2.imwrite(os.path.join(PREDICTIONS_PHASE2, filename), pred)
        
        pal_pred = Image.fromarray(pred).convert("P")
        pal_pred.putpalette(np.array(palette, dtype=np.uint8))
        pal_pred.save(os.path.join(PALETTE_PHASE2, filename))

print("Phase 2 Pseudo-label Generation Complete!")
