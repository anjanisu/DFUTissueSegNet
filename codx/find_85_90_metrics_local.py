import pandas as pd
import glob
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
res1 = glob.glob(os.path.join(BASE_DIR, 'Outputs', 'Results', '*.csv'))
res3 = glob.glob(os.path.join(BASE_DIR, 'Outputs', 'Results_Phase3', '*.csv'))
res4 = glob.glob(os.path.join(BASE_DIR, 'Outputs', 'Results_Phase4', '*.csv'))

def find_matches(csv_files, phase_name):
    for f in csv_files:
        df = pd.read_csv(f)
        df_valid = df[df['phase'] == 'valid']
        
        for idx, row in df_valid.iterrows():
            acc = row.get('accuracy', 0)
            f1 = row.get('fscore', 0)
            iou = row.get('iou_score', 0)
            
            # Identify values tracking between ~82% and ~92%
            if (0.80 <= acc <= 0.92) or (0.80 <= f1 <= 0.92):
                print(f"[{phase_name}] File: {os.path.basename(f)}, Epoch: {row['epoch']}")
                print(f"   Accuracy: {acc:.4f}, IoU: {iou:.4f}, F1 (DSC): {f1:.4f}")

print("Supervised matches:")
find_matches(res1, "Supervised")

print("\nSemi-Supervised Phase 1 matches:")
find_matches(res3, "Phase 1")

print("\nSemi-Supervised Phase 2 matches:")
find_matches(res4, "Phase 2")
