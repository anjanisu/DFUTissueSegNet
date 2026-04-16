import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Local paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
RESULTS_SUP_DIR = os.path.join(BASE_DIR, 'Outputs', 'Results')
RESULTS_PHASE3_DIR = os.path.join(BASE_DIR, 'Outputs', 'Results_Phase3')
PLOTS_DIR = os.path.join(BASE_DIR, 'Outputs', 'plots')

os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_plot(csv_path, save_name):
    print(f"Generating plot for {csv_path}")
    df = pd.read_csv(csv_path)
    
    train_df = df[df['phase'] == 'train']
    valid_df = df[df['phase'] == 'valid']
    
    # Identify the loss column (it's usually the 3rd column)
    loss_col = df.columns[2]
    
    epochs = train_df['epoch'].values
    
    store_train_loss = train_df[loss_col].values
    store_val_loss = valid_df[loss_col].values
    
    store_train_iou = train_df['iou_score'].values
    store_val_iou = valid_df['iou_score'].values
    
    store_train_dice = train_df['fscore'].values
    store_val_dice = valid_df['fscore'].values
    
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(epochs, store_train_loss, 'r', label='training')
    ax[0].plot(epochs, store_val_loss, 'b', label='validation')
    ax[0].set_title('Loss Curve')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()

    ax[1].plot(epochs, store_train_iou, 'r', label='training')
    ax[1].plot(epochs, store_val_iou, 'b', label='validation')
    ax[1].set_title('IoU Curve')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()

    ax[2].plot(epochs, store_train_dice, 'r', label='training')
    ax[2].plot(epochs, store_val_dice, 'b', label='validation')
    ax[2].set_title('Dice (F1) Curve')
    ax[2].set_xlabel('Epoch')
    ax[2].legend()

    fig.tight_layout()
    save_path = os.path.join(PLOTS_DIR, save_name)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved plot to {save_path}")

# Process Supervised CSV
try:
    sup_csv = glob.glob(os.path.join(RESULTS_SUP_DIR, '*.csv'))[0]
    generate_plot(sup_csv, 'supervised_training_curves.png')
except Exception as e:
    print(f"Could not process supervised CSV: {e}")

# Process Phase 3 CSVs (Semi-Supervised Phase 1)
try:
    phase3_csvs = glob.glob(os.path.join(RESULTS_PHASE3_DIR, '*.csv'))
    for phase3_csv in phase3_csvs:
        filename = os.path.basename(phase3_csv).replace('.csv', '.png')
        generate_plot(phase3_csv, f'semisupervised_phase1_{filename}')
except Exception as e:
    print(f"Could not process phase 3 CSVs: {e}")

# Process Phase 4 CSVs (Semi-Supervised Phase 2)
RESULTS_PHASE4_DIR = os.path.join(BASE_DIR, 'Outputs', 'Results_Phase4')
try:
    phase4_csvs = glob.glob(os.path.join(RESULTS_PHASE4_DIR, '*.csv'))
    for phase4_csv in phase4_csvs:
        filename = os.path.basename(phase4_csv).replace('.csv', '.png')
        generate_plot(phase4_csv, f'semisupervised_phase2_{filename}')
except Exception as e:
    print(f"Could not process phase 4 CSVs: {e}")

print("Plotting complete!")
