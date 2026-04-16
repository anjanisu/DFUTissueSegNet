import subprocess
import time

scripts = [
    "Codes/semisupervised_training_phase1_local.py",
    "Codes/pseudo_label_generation_phase2_local.py",
    "Codes/semisupervised_training_phase2_local.py",
    "Codes/generate_combined_loss_curves.py",
]

print("=== Starting Master Training Pipeline ===", flush=True)
start_time = time.time()

for script in scripts:
    print(f"\n---> Triggering: {script} at {time.strftime('%H:%M:%S')}", flush=True)
    subprocess.run([".\\.venv\\Scripts\\python", script], check=True)
    print(f"---> Successfully Completed: {script}", flush=True)

total_time = time.time() - start_time
print(f"\n=== Master Pipeline Finished in {total_time / 60:.2f} minutes ===", flush=True)
