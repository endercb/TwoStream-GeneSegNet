import sys
import os
import subprocess
import time
import re

def run_training(model_type, n_epochs=250):
    """Wrapper for training with best model tracking."""
    
    if model_type == "baseline":
        script = "/workspace/TwoStream-GeneSegNet/Code/GeneSeg_train.py"
        model_dir = "/workspace/TwoStream-GeneSegNet/TrainingOutput/baseline/models"
        log_file = "/workspace/TwoStream-GeneSegNet/TrainingOutput/baseline/training.log"
    else:
        script = "/workspace/TwoStream-GeneSegNet/Code/GeneSeg_train_twostream.py"
        model_dir = "/workspace/TwoStream-GeneSegNet/TrainingOutput/twostream/models"
        log_file = "/workspace/TwoStream-GeneSegNet/TrainingOutput/twostream/training.log"
    
    os.makedirs(model_dir, exist_ok=True)
    
    train_cmd = [
        "python", script,
        "--train",
        "--train_dir", "/workspace/TwoStream-GeneSegNet/Dataset/train",
        "--val_dir", "/workspace/TwoStream-GeneSegNet/Dataset/val",
        "--n_epochs", str(n_epochs),
        "--save_every", "10",
        "--save_each",
        "--save_model_dir", model_dir,
        "--use_gpu",
        "--all_channels",
        "--verbose"
    ]
    
    if model_type == "twostream":
        train_cmd.extend(["--cross_attn_layers", "4"])
    
    print(f"\n{'='*60}")
    print(f"Starting {model_type} training for {n_epochs} epochs")
    print(f"Log file: {log_file}")
    print(f"{'='*60}\n")
    
    with open(log_file, "w") as log_f:
        proc = subprocess.Popen(train_cmd, stdout=log_f, stderr=subprocess.STDOUT)
        proc.wait()
    
    print(f"\nTraining completed! Log saved to {log_file}")
    
    with open(log_file, "r") as f:
        content = f.read()
    
    epoch_losses = []
    for line in content.split("\n"):
        if "Epoch" in line and "Loss" in line and "Loss Test" in line:
            match = re.search(r'Epoch (\d+),.*?Loss ([\d.]+),.*?Loss Test ([\d.]+)', line)
            if match:
                epoch = int(match.group(1))
                train_loss = float(match.group(2))
                test_loss = float(match.group(3))
                epoch_losses.append((epoch, train_loss, test_loss))
    
    if epoch_losses:
        best_epoch = min(epoch_losses, key=lambda x: x[2])
        print(f"\nBest model: Epoch {best_epoch[0]} with test loss {best_epoch[2]:.4f}")
        
        best_model_path = os.path.join(model_dir, f"best_model_epoch_{best_epoch[0]}")
        
        for f in os.listdir(model_dir):
            if f"epoch_{best_epoch[0]}" in f:
                src = os.path.join(model_dir, f)
                dst = os.path.join(model_dir, f"best_model_{best_epoch[0]}.pth")
                import shutil
                shutil.copy(src, dst)
                print(f"Best model copied to: {dst}")
                break
        
        with open(os.path.join(model_dir, "training_summary.txt"), "w") as f:
            f.write(f"Best epoch: {best_epoch[0]}\n")
            f.write(f"Best test loss: {best_epoch[2]:.4f}\n")
            f.write(f"Train loss at best: {best_epoch[1]:.4f}\n\n")
            f.write("All epochs:\n")
            for e in epoch_losses:
                f.write(f"Epoch {e[0]}: train={e[1]:.4f}, test={e[2]:.4f}\n")
        
        print(f"Summary saved to {model_dir}/training_summary.txt")
    
    return epoch_losses

if __name__ == "__main__":
    print("="*60)
    print("GeneSegNet Training Wrapper")
    print("="*60)
    print("\n1. Baseline Model (250 epochs)")
    print("2. Two Encoder Model (250 epochs)")
    print("3. Both")
    
    choice = input("\nSelect option: ").strip()
    
    if choice in ["1", "3"]:
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL")
        print("="*60)
        run_training("baseline", 250)
    
    if choice in ["2", "3"]:
        print("\n" + "="*60)
        print("TRAINING TWO ENCODER MODEL")
        print("="*60)
        run_training("twostream", 250)
    
    print("\n" + "="*60)
    print("ALL TRAINING COMPLETE")
    print("="*60)