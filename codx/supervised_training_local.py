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
import json
import pandas as pd

DATASET_ROOT = os.path.join(BASE_DIR, 'DFUTissue', 'Labeled')
IMAGES_DIR = os.path.join(DATASET_ROOT, 'Padded', 'Images', 'TrainVal')
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, 'Padded', 'Annotations', 'TrainVal')
TEST_IMAGES_DIR = os.path.join(DATASET_ROOT, 'Padded', 'Images', 'Test')
TEST_ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, 'Padded', 'Annotations', 'Test')
SAVE_DIR_ROOT = os.path.join(BASE_DIR, 'Outputs')
RESULTS_DIR = os.path.join(SAVE_DIR_ROOT, 'Results')

os.makedirs(SAVE_DIR_ROOT, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class Dataset(BaseDataset):
    def __init__(self, list_IDs, images_dir, masks_dir, augmentation=None, preprocessing=None, to_categorical=False, resize=(False, (256, 256)), n_classes=6, default_img=None, default_mask=None):
        self.ids = list_IDs
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.to_categorical = to_categorical
        self.resize = resize
        self.n_classes = n_classes
        self.default_img = default_img
        self.default_mask = default_mask

    def __getitem__(self, i):
        try:
            image = cv2.imread(self.images_fps[i])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(self.masks_fps[i], 0)
        except Exception as e:
            image = self.default_img
            mask = self.default_mask

        if self.resize[0]:
            image = cv2.resize(image, self.resize[1], interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, self.resize[1], interpolation=cv2.INTER_NEAREST)

        mask = np.expand_dims(mask, axis=-1)
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.to_categorical:
            mask = torch.from_numpy(mask)
            mask = F.one_hot(mask.long(), num_classes=self.n_classes)
            mask = mask.type(torch.float32).numpy()
            mask = np.squeeze(mask)
            mask = np.moveaxis(mask, -1, 0)
        return image, mask

    def __len__(self):
        return len(self.ids)

def get_training_augmentation():
    train_transform = [
        albu.OneOf([albu.HorizontalFlip(p=0.5), albu.VerticalFlip(p=0.5)], p=0.8),
        albu.OneOf([
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.1, border_mode=0),
            albu.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=0.1, border_mode=0),
            albu.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=0.2, border_mode=0),
        ], p=0.9),
        albu.OneOf([
            albu.Perspective(p=0.2),
            albu.GaussNoise(p=0.2),
            albu.Sharpen(p=0.2),
            albu.Blur(blur_limit=3, p=0.2),
            albu.MotionBlur(blur_limit=3, p=0.2),
        ], p=0.5),
        albu.OneOf([
            albu.CLAHE(p=0.25),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
            albu.RandomGamma(p=0.25),
            albu.HueSaturationValue(p=0.25),
        ], p=0.3),
    ]
    return albu.Compose(train_transform, p=0.9)

def get_validation_augmentation():
    return albu.Compose([])

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [albu.Lambda(image=preprocessing_fn), albu.Lambda(image=to_tensor, mask=to_tensor)]
    return albu.Compose(_transform)

# Parameters
BASE_MODEL = 'MiT+pscse'
ENCODER = 'mit_b3'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 4 
n_classes = 4
ACTIVATION = 'sigmoid'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0001
EPOCHS = 100 # Increased for a full run
WEIGHT_DECAY = 1e-5
SAVE_BEST_MODEL = True
EARLY_STOP = True
PATIENCE = 10

def read_names(txt_file, ext=".png"):
    with open(txt_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return [name + ext for name in names]

def save(model_path, epoch, model_state_dict, optimizer_state_dict):
    state = {'epoch': epoch + 1, 'state_dict': deepcopy(model_state_dict), 'optimizer': deepcopy(optimizer_state_dict)}
    torch.save(state, model_path)

# Main
print(f"Using device: {DEVICE}")
model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=n_classes, activation=ACTIVATION, decoder_attention_type='pscse')
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
model.to(DEVICE)

optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=10, min_lr=0.00001)

list_IDs_train = read_names(os.path.join(DATASET_ROOT, 'labeled_train_names.txt'))
list_IDs_val = read_names(os.path.join(DATASET_ROOT, 'labeled_val_names.txt'))
list_IDs_test = read_names(os.path.join(DATASET_ROOT, 'test_names.txt'))

print(f"Training: {len(list_IDs_train)}, Val: {len(list_IDs_val)}, Test: {len(list_IDs_test)}")

DEFAULT_IMG_TRAIN = cv2.imread(os.path.join(IMAGES_DIR, list_IDs_train[0]))[:,:,::-1]
DEFAULT_MASK_TRAIN = cv2.imread(os.path.join(ANNOTATIONS_DIR, list_IDs_train[0]), 0)
DEFAULT_IMG_VAL = cv2.imread(os.path.join(IMAGES_DIR, list_IDs_val[0]))[:,:,::-1]
DEFAULT_MASK_VAL = cv2.imread(os.path.join(ANNOTATIONS_DIR, list_IDs_val[0]), 0)

checkpoint_loc = os.path.join(SAVE_DIR_ROOT, 'checkpoints', f"{BASE_MODEL}_{ENCODER}_sup")
os.makedirs(checkpoint_loc, exist_ok=True)

train_dataset = Dataset(list_IDs_train, IMAGES_DIR, ANNOTATIONS_DIR, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), to_categorical=True, n_classes=n_classes, default_img=DEFAULT_IMG_TRAIN, default_mask=DEFAULT_MASK_TRAIN)
valid_dataset = Dataset(list_IDs_val, IMAGES_DIR, ANNOTATIONS_DIR, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), to_categorical=True, n_classes=n_classes, default_img=DEFAULT_IMG_VAL, default_mask=DEFAULT_MASK_VAL)
test_dataset = Dataset(list_IDs_test, TEST_IMAGES_DIR, TEST_ANNOTATIONS_DIR, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), to_categorical=True, n_classes=n_classes)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

all_metrics = [
    metrics.IoU(threshold=0.5),
    metrics.Fscore(threshold=0.5),
    metrics.Accuracy(threshold=0.5),
    metrics.Precision(threshold=0.5),
    metrics.Recall(threshold=0.5),
]

train_epoch = smp.utils.train.TrainEpoch(
    model, 
    loss=base.SumOfLosses(losses.DiceLoss(), losses.FocalLoss()), 
    metrics=all_metrics, 
    optimizer=optimizer, 
    device=DEVICE, 
    verbose=True
)

valid_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=base.SumOfLosses(losses.DiceLoss(), losses.FocalLoss()), 
    metrics=all_metrics, 
    device=DEVICE, 
    verbose=True
)

best_vloss = 1e6
cnt_patience = 0
history = []

for epoch in range(EPOCHS):
    print(f"\nEpoch: {epoch}")
    train_logs = train_epoch.run(train_loader)
    valid_logs = valid_epoch.run(valid_loader)
    
    log_entry = {'epoch': epoch, 'phase': 'train'}
    log_entry.update(train_logs)
    history.append(log_entry)
    
    log_entry_val = {'epoch': epoch, 'phase': 'valid'}
    log_entry_val.update(valid_logs)
    history.append(log_entry_val)
    
    val_loss = valid_logs['dice_loss + focal_loss']
    if best_vloss > val_loss:
        best_vloss = val_loss
        print(f"Validation loss reduced. Saving model.")
        save(os.path.join(checkpoint_loc, 'best_model.pth'), epoch, model.state_dict(), optimizer.state_dict())
        cnt_patience = 0
    else:
        cnt_patience += 1
    
    scheduler.step(val_loss)
    
    # Save training history incrementally
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(RESULTS_DIR, 'training_metrics_history.csv'), index=False)
    
    if EARLY_STOP and cnt_patience >= PATIENCE:
        print("Early stopping.")
        break

# Save training history
history_df = pd.DataFrame(history)
history_df.to_csv(os.path.join(RESULTS_DIR, 'training_metrics_history.csv'), index=False)

print("\nTraining complete. Starting final evaluation on test set...")

# Final Evaluation on Test Set
model.load_state_dict(torch.load(os.path.join(checkpoint_loc, 'best_model.pth'))['state_dict'])
test_epoch = smp.utils.train.ValidEpoch(
    model, 
    loss=base.SumOfLosses(losses.DiceLoss(), losses.FocalLoss()), 
    metrics=all_metrics, 
    device=DEVICE, 
    verbose=True
)

test_logs = test_epoch.run(test_loader)

# Map metric names for clarity as requested
final_results = {
    "Accuracy": test_logs['accuracy'],
    "Precision": test_logs['precision'],
    "Recall": test_logs['recall'],
    "F1-score (DSC)": test_logs['fscore'],
    "IoU (Jaccard)": test_logs['iou_score'],
    "Total Loss": test_logs['dice_loss + focal_loss']
}

with open(os.path.join(RESULTS_DIR, 'final_test_results.json'), 'w') as f:
    json.dump(final_results, f, indent=4)

with open(os.path.join(RESULTS_DIR, 'final_test_summary.txt'), 'w') as f:
    for k, v in final_results.items():
        line = f"{k}: {v:.4f}\n"
        print(line, end='')
        f.write(line)

print(f"\nResults saved to {RESULTS_DIR}")
