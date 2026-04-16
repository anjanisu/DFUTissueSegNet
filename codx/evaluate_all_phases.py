import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import cv2
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
CODES_DIR = os.path.join(BASE_DIR, 'Codes')
if CODES_DIR not in sys.path:
    sys.path.insert(0, CODES_DIR)

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.utils import metrics, losses, base
import torch.nn.functional as F

DATASET_ROOT = os.path.join(BASE_DIR, 'DFUTissue')
LABELED_DIR = os.path.join(DATASET_ROOT, 'Labeled', 'Padded')
OUTPUT_DIR_ROOT = os.path.join(BASE_DIR, 'Outputs')

class Dataset(BaseDataset):
    def __init__(self, data_pairs, preprocessing=None, resize=(False, (256, 256)), n_classes=4):
        self.data_pairs = data_pairs 
        self.preprocessing = preprocessing
        self.resize = resize
        self.n_classes = n_classes

    def __getitem__(self, i):
        img_path, mask_path = self.data_pairs[i]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, 0)
        
        if self.resize[0]:
            image = cv2.resize(image, self.resize[1], interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, self.resize[1], interpolation=cv2.INTER_NEAREST)

        mask = np.expand_dims(mask, axis=-1)
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        mask = torch.from_numpy(mask)
        mask = F.one_hot(mask.long(), num_classes=self.n_classes)
        mask = mask.type(torch.float32).numpy()
        mask = np.squeeze(mask)
        mask = np.moveaxis(mask, -1, 0)
        return image, mask

    def __len__(self): return len(self.data_pairs)

def to_tensor(x, **kwargs): return x.transpose(2, 0, 1).astype('float32')
def get_preprocessing(preprocessing_fn):
    return albu.Compose([albu.Lambda(image=preprocessing_fn), albu.Lambda(image=to_tensor, mask=to_tensor)])

def read_names(txt_file, ext=".png"):
    with open(txt_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return [name + ext for name in names]

sup_test_names = read_names(os.path.join(LABELED_DIR, '..', 'test_names.txt'))
test_pairs = [(os.path.join(LABELED_DIR, 'Images', 'Test', x), os.path.join(LABELED_DIR, 'Annotations', 'Test', x)) for x in sup_test_names]

ENCODER = 'mit_b3'
ENCODER_WEIGHTS = 'imagenet'
n_classes = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
test_dataset = Dataset(test_pairs, preprocessing=get_preprocessing(preprocessing_fn), n_classes=n_classes)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

all_metrics = [
    metrics.Accuracy(threshold=0.5),
    metrics.IoU(threshold=0.5),
    metrics.Precision(threshold=0.5),
    metrics.Recall(threshold=0.5),
    metrics.Fscore(threshold=0.5), # DSC
]

dce_loss = losses.DynamicCEAndSCELoss()
total_loss = base.HybridLoss(losses.DiceLoss(), losses.FocalLoss(), dce_loss, [1.0, 1.0, 1.0])

model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=n_classes, activation='sigmoid', decoder_attention_type='pscse')
model.to(DEVICE)

def evaluate_model(checkpoint_path, phase_name):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Not found {checkpoint_path}")
        return
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE)['state_dict'])
    test_epoch = smp.utils.train.ValidEpoch(model, loss=total_loss, metrics=all_metrics, device=DEVICE, verbose=False)
    logs = test_epoch.run(test_loader)
    print(f"--- {phase_name} ---")
    print(f"Accuracy: {logs['accuracy']:.4f}")
    print(f"IoU: {logs['iou_score']:.4f}")
    print(f"Precision: {logs['precision']:.4f}")
    print(f"Recall: {logs['recall']:.4f}")
    print(f"DSC (F1): {logs['fscore']:.4f}\n")

print("Evaluating all phases on Test Set with complete metrics...\n")

evaluate_model(os.path.join(OUTPUT_DIR_ROOT, 'checkpoints', 'MiT+pscse_mit_b3_sup', 'best_model.pth'), "Supervised Training")
evaluate_model(os.path.join(OUTPUT_DIR_ROOT, 'checkpoints_Phase3', 'MiT+pscse_padded_mit_b3_run_2_selfSupervised', 'best_model.pth'), "Semi-Supervised Phase 1 (Run 2)")
evaluate_model(os.path.join(OUTPUT_DIR_ROOT, 'checkpoints_Phase4', 'MiT+pscse_padded_mit_b3_run_2_selfSupervised_phase2', 'best_model.pth'), "Semi-Supervised Phase 2 (Run 2)")
