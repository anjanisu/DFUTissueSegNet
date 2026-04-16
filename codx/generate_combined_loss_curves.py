import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Local paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
RESULTS_PHASE3_DIR = os.path.join(BASE_DIR, 'Outputs', 'Results_Phase3')
RESULTS_PHASE4_DIR = os.path.join(BASE_DIR, 'Outputs', 'Results_Phase4')
PLOTS_DIR = os.path.join(BASE_DIR, 'Outputs', 'plots')

os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_combined_plot(csv_dir, save_name, title_prefix):
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    csv_files.sort() # Ensure consistent order like run_0, run_1...
    
    if not csv_files:
        print(f"No CSVs found in {csv_dir}")
        return

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{title_prefix} - All Runs Combined (Train=Dashed, Val=Solid)', fontsize=16)

    colors = plt.cm.get_cmap('tab10', len(csv_files))

    for idx, csv_path in enumerate(csv_files):
        run_name = os.path.basename(csv_path).replace('_training_metrics_history.csv', '')
        df = pd.read_csv(csv_path)
        
        train_df = df[df['phase'] == 'train']
        valid_df = df[df['phase'] == 'valid']
        
        loss_col = df.columns[2]
        
        epochs = train_df['epoch'].values
        
        store_train_loss = train_df[loss_col].values
        store_val_loss = valid_df[loss_col].values
        store_train_iou = train_df['iou_score'].values
        store_val_iou = valid_df['iou_score'].values
        store_train_dice = train_df['fscore'].values
        store_val_dice = valid_df['fscore'].values
        
        c = colors(idx)
        
        # Plot Loss
        ax[0].plot(epochs, store_train_loss, '--', color=c, alpha=0.5)
        ax[0].plot(epochs, store_val_loss, '-', color=c, label=f'{run_name} (val)')
        
        # Plot IoU
        ax[1].plot(epochs, store_train_iou, '--', color=c, alpha=0.5)
        ax[1].plot(epochs, store_val_iou, '-', color=c, label=f'{run_name} (val)')
        
        # Plot Dice
        ax[2].plot(epochs, store_train_dice, '--', color=c, alpha=0.5)
        ax[2].plot(epochs, store_val_dice, '-', color=c, label=f'{run_name} (val)')

    ax[0].set_title('Loss')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()

    ax[1].set_title('IoU')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()

    ax[2].set_title('Dice (F1)')
    ax[2].set_xlabel('Epoch')
    ax[2].legend()

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(PLOTS_DIR, save_name)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved combined plot to {save_path}")

print("Generating Combined Phase 1 Plot...")
generate_combined_plot(RESULTS_PHASE3_DIR, 'semisupervised_phase1_combined.png', 'Semi-Supervised Phase 1')

print("Generating Combined Phase 2 Plot...")
generate_combined_plot(RESULTS_PHASE4_DIR, 'semisupervised_phase2_combined.png', 'Semi-Supervised Phase 2')
