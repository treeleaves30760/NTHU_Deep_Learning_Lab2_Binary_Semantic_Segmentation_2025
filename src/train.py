from models.unet import UNet, UNetLarge, UNetXL
from evaluate import evaluate
from models.resnet34_unet import ResNet34_UNet
import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from oxford_pet import load_dataset


# Add DICE loss function
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)

        # Smoothing
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / \
            (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


def train(args):
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create storage directories
    os.makedirs('./saved_models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    # Set up TensorBoard
    log_dir = f'./logs/{time.strftime("%Y%m%d_%H%M%S")}_{args.model_type}'
    writer = SummaryWriter(log_dir)

    # Create CSV log file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f'./logs/{timestamp}_{args.model_type}_training_log.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train_Loss', 'Val_Dice', 'Learning_Rate'])

    # Initialize record lists
    train_losses = []
    val_dices = []
    learning_rates = []

    # Load dataset
    train_dataset = load_dataset(args.data_path, mode="train")
    val_dataset = load_dataset(args.data_path, mode="valid")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    if args.model_type.lower() == 'unet':
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
    elif args.model_type.lower() == 'unet_large':
        model = UNetLarge(n_channels=3, n_classes=1, bilinear=True)
    elif args.model_type.lower() == 'unet_xl':
        model = UNetXL(n_channels=3, n_classes=1, bilinear=True)
    elif args.model_type.lower() == 'resnet34_unet':
        model = ResNet34_UNet(n_classes=1, pretrained=True, bilinear=True)
    else:
        raise ValueError(
            "Unsupported model type. Please choose 'unet', 'unet_large', 'unet_xl', or 'resnet34_unet'")

    model = model.to(device)

    # Define loss function and optimizer
    criterion = DiceLoss(smooth=1.0)  # Use DICE loss function

    # Use AdamW optimizer for better weight decay
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # Use OneCycleLR learning rate scheduler for better convergence performance
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
                                              total_steps=total_steps,
                                              pct_start=0.3,  # 30% time for warm-up
                                              div_factor=25,  # Initial learning rate = max_lr/div_factor
                                              final_div_factor=1000)  # Final learning rate = max_lr/(div_factor*final_div_factor)

    # Training loop
    best_dice = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        batch_losses = []  # Record losses for each batch

        # Training
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(device, dtype=torch.float32)
                true_masks = batch['mask'].to(device, dtype=torch.float32)

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, true_masks)

                # Backward propagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Record current batch loss
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_losses.append(batch_loss)

                # Record to TensorBoard (every 100 batches)
                if batch_idx % 100 == 0:
                    writer.add_scalar('Batch Loss', batch_loss,
                                      epoch * len(train_loader) + batch_idx)
                    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'],
                                      epoch * len(train_loader) + batch_idx)

                pbar.update(1)
                pbar.set_postfix(
                    {'loss': batch_loss, 'lr': optimizer.param_groups[0]['lr']})

        # Calculate average training loss
        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        val_dice = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        # Record to lists
        train_losses.append(float(avg_train_loss))
        val_dices.append(float(val_dice))
        learning_rates.append(float(current_lr))

        # Record to TensorBoard
        writer.add_scalar('Training Loss/epoch', avg_train_loss, epoch)
        writer.add_scalar('Validation Dice/epoch', val_dice, epoch)
        writer.add_scalar('Learning Rate/epoch', current_lr, epoch)

        # Record to CSV
        csv_writer.writerow([epoch+1, avg_train_loss, val_dice, current_lr])
        csv_file.flush()  # Write immediately, don't wait for buffer

        print(
            f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_train_loss:.4f}, Validation Dice: {val_dice:.4f}, LR: {current_lr:.6f}')

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            model_path = f'./saved_models/{time.strftime("%Y%m%d_%H%M%S")}_{args.model_type}_best_model_{best_dice:.4f}.pth'
            torch.save(model.state_dict(), model_path)
            print(
                f'Saved new best model, Dice: {best_dice:.4f}, Path: {model_path}')

    # Save final model
    final_model_path = f'./saved_models/{args.model_type}_final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(
        f'Training completed. Best validation Dice: {best_dice:.4f}, Final model path: {final_model_path}')

    # Close CSV file and TensorBoard
    csv_file.close()
    writer.close()

    # Plot and save learning curves
    plot_learning_curves(train_losses, val_dices,
                         learning_rates, args.model_type)

    return {
        'train_losses': train_losses,
        'val_dices': val_dices,
        'learning_rates': learning_rates,
        'best_dice': best_dice
    }


def plot_learning_curves(train_losses, val_dices, learning_rates, model_type):
    """Plot learning curves and save as image"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Ensure data is on CPU and convert to NumPy arrays
    if torch.is_tensor(train_losses):
        train_losses = train_losses.cpu().numpy()
    if torch.is_tensor(val_dices):
        val_dices = val_dices.cpu().numpy()
    if torch.is_tensor(learning_rates):
        learning_rates = learning_rates.cpu().numpy()

    # Create chart
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot training loss and validation Dice
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'bo-', label='Training Loss')
    ax1.plot(epochs, val_dices, 'ro-', label='Validation Dice Coefficient')
    ax1.set_title(f'{model_type} Model Training Curves')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss / Dice Coefficient')
    ax1.legend()
    ax1.grid(True)

    # Plot learning rate
    ax2.plot(epochs, learning_rates, 'go-', label='Learning Rate')
    ax2.set_title('Learning Rate Changes')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Learning Rate')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f'./logs/{timestamp}_{model_type}_learning_curves.png')
    plt.close()
    print(
        f'Learning curves saved to: ./logs/{timestamp}_{model_type}_learning_curves.png')


def get_args():
    parser = argparse.ArgumentParser(
        description='Train UNet with image and target mask')
    parser.add_argument('--data_path', type=str,
                        help='Path to input data', default='./dataset/oxford-iiit-pet')
    parser.add_argument('--epochs', '-e', type=int,
                        default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=16, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', type=float,
                        default=3e-4, help='Learning rate')
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'unet_large',
                                 'unet_xl', 'resnet34_unet'],
                        help='Model architecture (unet, unet_large, unet_xl, or resnet34_unet)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    results = train(args)
    print("Training finished!")
