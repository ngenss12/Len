import torch
import os
import random
import json
import numpy as np
import matplotlib.pyplot as plt
from model import ResNet50WithRT
from dataload import FinalDataset
from torch.utils.data import DataLoader
from train import train_model
from metrics import evaluate_model, print_results, plot_confusion_matrix
from datetime import datetime


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _make_unique_dir(path):
    if not os.path.exists(path):
        return path
    idx = 1
    while True:
        candidate = f"{path}_r{idx}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def save_training_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        return

    # 1) Loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "history_loss.png"), dpi=160)
    plt.close()

    # 2) F1 curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_f1"], label="train_f1")
    plt.plot(epochs, history["val_f1"], label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 (%)")
    plt.title("Training vs Validation F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "history_f1.png"), dpi=160)
    plt.close()

    # 3) Accuracy curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "history_acc.png"), dpi=160)
    plt.close()

    with open(os.path.join(output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def main():
    # Configuration
    use_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if use_cuda else 0
    requested_gpu = int(os.getenv("GPU_ID", "0"))

    if use_cuda and gpu_count > 0:
        if requested_gpu < 0 or requested_gpu >= gpu_count:
            print(f"GPU_ID={requested_gpu} tidak valid. Fallback ke GPU 0.")
            requested_gpu = 0
        gpu_id = requested_gpu
        device = torch.device(f"cuda:{gpu_id}")
    else:
        gpu_id = -1
        device = torch.device("cpu")

    config = {
        'seed': int(os.getenv("SEED", "42")),
        'epochs': int(os.getenv("EPOCHS", "250")),
        'learning_rate': float(os.getenv("LR", "0.0001")),
        'early_stop_patience': int(os.getenv("EARLY_STOP_PATIENCE", "0")),
        'f1_loss_weight': float(os.getenv("F1_LOSS_WEIGHT", "0.0")),
        'freeze_backbone': os.getenv("FREEZE_BACKBONE", "0") == "1",
        'num_workers': 8,
        'batch_size': int(os.getenv("BATCH_SIZE", "32")),
        'device': str(device),
        'gpu_id': gpu_id,
        'resume_from': os.getenv("RESUME_FROM", "").strip(),
    }

    seed_everything(config['seed'])
    previous_best_val_f1 = 0.0

    model_type = 'resnet50'
    output_dir_base = f"experiments/{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lr{config['learning_rate']}"
    output_dir = _make_unique_dir(output_dir_base)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using device: {config['device']}")
    if use_cuda:
        print(f"Detected GPUs: {gpu_count}")
        for i in range(gpu_count):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
        print(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
        torch.backends.cudnn.benchmark = True
    print(f"Config: epochs={config['epochs']}, lr={config['learning_rate']}, "
          f"batch_size={config['batch_size']}, early_stop_patience={config['early_stop_patience']}, "
          f"f1_loss_weight={config['f1_loss_weight']}, freeze_backbone={config['freeze_backbone']}, "
          f"resume_from={config['resume_from'] or 'None'}")

    # Load augmented training data
    print("\nLoading datasets...")
    train_balanced = FinalDataset(csv_path="final/train.csv")
    val_balanced = FinalDataset(csv_path="final/val.csv")

    # Create data loaders
    g = torch.Generator()
    g.manual_seed(config['seed'])
    train_loader = DataLoader(
        train_balanced,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=use_cuda,
        worker_init_fn=_seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_balanced,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=use_cuda,
        worker_init_fn=_seed_worker,
    )

    # 5. Build model
    print("\nTraining ResNet50 model...")
    model = ResNet50WithRT(rt_dim=8, num_classes=4, pretrained=True)
    if config['freeze_backbone']:
        # Freeze most pretrained CNN layers, fine-tune only top block + custom heads.
        for p in model.resnet.parameters():
            p.requires_grad = False
        for p in model.resnet[7].parameters():  # layer4
            p.requires_grad = True
    model = model.to(config['device'])

    # Optional resume from previous best checkpoint (weights only).
    if config['resume_from']:
        resume_path = config['resume_from']
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"RESUME_FROM checkpoint not found: {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' not in checkpoint:
            raise KeyError(f"Checkpoint missing 'model_state_dict': {resume_path}")
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        hist = checkpoint.get('history', {})
        prev_vals = hist.get('val_f1', []) if isinstance(hist, dict) else []
        if isinstance(prev_vals, list) and len(prev_vals) > 0:
            previous_best_val_f1 = float(max(prev_vals))
        print(f"Resumed model weights from: {resume_path}")
        print(f"Previous best Val F1 from checkpoint: {previous_best_val_f1:.2f}%")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 6. Train model
    print("\nTraining model...")
    model, history = train_model(
        model, train_loader, val_loader,
        device=config['device'],
        epochs=config['epochs'],
        initial_lr=config['learning_rate'],
        early_stop_patience=config['early_stop_patience'],
        f1_loss_weight=config['f1_loss_weight'],
        previous_best_val_f1=previous_best_val_f1,
    )

    # 7. Save model
    print("\nSaving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, f'{output_dir}/model.pth')

    # 8. Evaluate on validation set
    print("\n6. Final evaluation...")
    results = evaluate_model(model, val_loader, config['device'])

    print_results(results)
    plot_confusion_matrix(results['confusion_matrix'], output_dir=output_dir)

    print("Training complete!")
    print(f'Best validation F1-score: {max(history["val_f1"]):.2f}%')

    with open(f'{output_dir}/results.txt', 'w') as f:
        print(results, file=f)
        print(history, file=f)

    save_training_plots(history, output_dir)
    print(f"Saved training plots to: {output_dir}")

    return model, history, results

if __name__ == '__main__':
    model, history, results = main()
