import torch
import os
from model import ResNet50WithRT
from dataload import FinalDataset
from torch.utils.data import DataLoader
from train import train_model
from metrics import evaluate_model, print_results, plot_confusion_matrix
from datetime import datetime

def main():
    # Configuration
    config = {
        'epochs': 100,
        'learning_rate': 0.0001,
        'num_workers': 8,
        'batch_size': 64,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    model_type = 'resnet50'
    output_dir = f"experiments/{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_lr{config['learning_rate']}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Using device: {config['device']}")

    # Load augmented training data
    print("\nLoading datasets...")
    train_balanced = FinalDataset(csv_path="final/train.csv")
    val_balanced = FinalDataset(csv_path="final/val.csv")

    # Create data loaders
    train_loader = DataLoader(
        train_balanced,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_balanced,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    # 5. Build model
    print("\nTraining ResNet50 model...")
    model = ResNet50WithRT(rt_dim=8, num_classes=4, pretrained=True)
    model = model.to(config['device'])

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
        initial_lr=config['learning_rate']
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

    return model, history, results

if __name__ == '__main__':
    model, history, results = main()
