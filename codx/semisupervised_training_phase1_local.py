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
from segmentation_models_pytorch.utils import metrics, losses, base
import random
from copy import deepcopy
import torch.nn.functional as F
import json
import pandas as pd

DATASET_ROOT = os.path.join(BASE_DIR, 'DFUTissue')
LABELED_DIR = os.path.join(DATASET_ROOT, 'Labeled', 'Padded')
UNLABELED_DIR = os.path.join(DATASET_ROOT, 'Unlabeled')

OUTPUT_DIR_ROOT = os.path.join(BASE_DIR, 'Outputs')
RESULTS_DIR = os.path.join(OUTPUT_DIR_ROOT, 'Results_Phase3')
os.makedirs(RESULTS_DIR, exist_ok=True)
CHECKPOINT_ROOT = os.path.join(OUTPUT_DIR_ROOT, 'checkpoints_Phase3')
os.makedirs(CHECKPOINT_ROOT, exist_ok=True)

class Dataset(BaseDataset):
    def __init__(self, data_pairs, augmentation=None, preprocessing=None, to_categorical=False, resize=(False, (256, 256)), n_classes=4):
        self.data_pairs = data_pairs # list of (img_path, mask_path)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.to_categorical = to_categorical
        self.resize = resize
        self.n_classes = n_classes

    def __getitem__(self, i):
        img_path, mask_path = self.data_pairs[i]
        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, 0)
        except Exception as e:
            pass

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
        return len(self.data_pairs)

def get_training_augmentation():
    train_transform = [
        albu.OneOf([albu.HorizontalFlip(p=0.5), albu.VerticalFlip(p=0.5)], p=0.8),
        albu.OneOf([
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0, p=0.1, border_mode=0),
            albu.ShiftScaleRotate(scale_limit=0, rotate_limit=30, shift_limit=0, p=0.1, border_mode=0),
            albu.ShiftScaleRotate(scale_limit=0, rotate_limit=0, shift_limit=0.1, p=0.6, border_mode=0),
            albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=30, shift_limit=0.1, p=0.2, border_mode=0),
        ], p=0.9),
    ]
    return albu.Compose(train_transform, p=0.9)

def get_validation_augmentation():
    return albu.Compose([])

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    _transform = [albu.Lambda(image=preprocessing_fn), albu.Lambda(image=to_tensor, mask=to_tensor)]
    return albu.Compose(_transform)

def read_names(txt_file, ext=".png"):
    with open(txt_file, "r") as f:
        names = [line.strip() for line in f if line.strip()]
    return [name + ext for name in names]

def save(model_path, epoch, model_state_dict, optimizer_state_dict):
    state = {'epoch': epoch + 1, 'state_dict': deepcopy(model_state_dict), 'optimizer': deepcopy(optimizer_state_dict)}
    torch.save(state, model_path)

# Parameters
BASE_MODEL = 'MiT+pscse'
ENCODER = 'mit_b3'
ENCODER_WEIGHTS = 'imagenet'
BATCH_SIZE = 4 
n_classes = 4
ACTIVATION = 'sigmoid'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.0001
EPOCHS = 10
WEIGHT_DECAY = 1e-5
EARLY_STOP = True
PATIENCE = 10

# Load Names
sup_train_names = read_names(os.path.join(LABELED_DIR, '..', 'labeled_train_names.txt'))
sup_val_names = read_names(os.path.join(LABELED_DIR, '..', 'labeled_val_names.txt'))
sup_test_names = read_names(os.path.join(LABELED_DIR, '..', 'test_names.txt'))

unsup_names = os.listdir(UNLABELED_DIR)

# Validation Pairs
val_pairs = [(os.path.join(LABELED_DIR, 'Images', 'TrainVal', x), os.path.join(LABELED_DIR, 'Annotations', 'TrainVal', x)) for x in sup_val_names]
test_pairs = [(os.path.join(LABELED_DIR, 'Images', 'Test', x), os.path.join(LABELED_DIR, 'Annotations', 'Test', x)) for x in sup_test_names]

all_metrics = [
    metrics.IoU(threshold=0.5),
    metrics.Fscore(threshold=0.5),
    metrics.Accuracy(threshold=0.5),
]

dice_loss = losses.DiceLoss()
focal_loss = losses.FocalLoss()
dce_loss = losses.DynamicCEAndSCELoss()
weight_factor = [1.0, 1.0, 1.0]

total_loss = base.HybridLoss(dice_loss, focal_loss, dce_loss, weight_factor)

n_runs = 5
seeds = [random.randint(0, 5000) for _ in range(n_runs)]

best_val_losses_across_runs = []

for run, seed in enumerate(seeds):
    print(f'\n=========================== run {run+1}/{n_runs} (seed {seed}) ============================')
    model_name = BASE_MODEL + '_padded_' + ENCODER + f'_run_{run}_selfSupervised'
    print(f"Model Name: {model_name}")
    
    # create segmentation model with pretrained encoder
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=ENCODER_WEIGHTS, classes=n_classes, activation=ACTIVATION, decoder_attention_type='pscse')
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    model.to(DEVICE)

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=10, min_lr=0.00001)

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Pick 50 unlabeled
    unsup_names_IDs = unsup_names.copy()
    random.shuffle(unsup_names_IDs)
    unsup_names_IDs = unsup_names_IDs[:50]

    # Training Pairs
    sup_train_pairs = [(os.path.join(LABELED_DIR, 'Images', 'TrainVal', x), os.path.join(LABELED_DIR, 'Annotations', 'TrainVal', x)) for x in sup_train_names]
    unsup_train_pairs = [(os.path.join(UNLABELED_DIR, x), os.path.join(OUTPUT_DIR_ROOT, 'Predictions', 'Phase1', x)) for x in unsup_names_IDs]
    train_pairs = sup_train_pairs + unsup_train_pairs
    
    # Save the selected 50 for Phase 4 tracking
    with open(os.path.join(RESULTS_DIR, f'run_{run}_unsup_train.txt'), "w") as f:
        for name in unsup_names_IDs: f.write(name + "\n")

    checkpoint_loc = os.path.join(CHECKPOINT_ROOT, model_name)
    os.makedirs(checkpoint_loc, exist_ok=True)

    train_dataset = Dataset(train_pairs, augmentation=get_training_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), to_categorical=True, n_classes=n_classes)
    valid_dataset = Dataset(val_pairs, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), to_categorical=True, n_classes=n_classes)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    train_epoch = smp.utils.train.TrainEpoch(model, loss=total_loss, metrics=all_metrics, optimizer=optimizer, device=DEVICE, verbose=True)
    valid_epoch = smp.utils.train.ValidEpoch(model, loss=total_loss, metrics=all_metrics, device=DEVICE, verbose=True)

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
        
        val_loss = valid_logs[list(valid_logs.keys())[0]] # HybridLoss
        
        if best_vloss > val_loss:
            best_vloss = val_loss
            print(f"Validation loss reduced to {best_vloss:.4f}. Saving model.")
            save(os.path.join(checkpoint_loc, 'best_model.pth'), epoch, model.state_dict(), optimizer.state_dict())
            cnt_patience = 0
        else:
            cnt_patience += 1
        
        scheduler.step(val_loss)
            
    best_val_losses_across_runs.append((run, best_vloss, checkpoint_loc))
    
    # Save training history for this run
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(RESULTS_DIR, f'run_{run}_training_metrics_history.csv'), index=False)

# Evaluate the best run on test set
best_run = min(best_val_losses_across_runs, key=lambda x: x[1])
print(f"\nTraining for 5 runs completed. Best Run: {best_run[0]} with validation loss {best_run[1]}")

test_dataset = Dataset(test_pairs, augmentation=get_validation_augmentation(), preprocessing=get_preprocessing(preprocessing_fn), to_categorical=True, n_classes=n_classes)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

model.load_state_dict(torch.load(os.path.join(best_run[2], 'best_model.pth'))['state_dict'])
test_epoch = smp.utils.train.ValidEpoch(model, loss=total_loss, metrics=all_metrics, device=DEVICE, verbose=True)
test_logs = test_epoch.run(test_loader)

with open(os.path.join(BASE_DIR, 'output.txt'), 'a') as f:
    f.write(f"\n# Phase 3: Semi-Supervised Training Phase 1\n")
    f.write(f"- Tested 5 iterations of randomly selected 50 pseudo-labeled images with 78 labeled images.\n")
    f.write(f"- Best Run: {best_run[0]} with validation loss: {best_run[1]:.4f}\n")
    f.write(f"- Evaluation of Phase 3 on Test Set:\n")
    f.write(f"     Accuracy: {test_logs['accuracy']:.4f}\n")
    f.write(f"     F1-score: {test_logs['fscore']:.4f}\n")
    f.write(f"     IoU: {test_logs['iou_score']:.4f}\n")

print("\nResults successfully appended to output.txt")

