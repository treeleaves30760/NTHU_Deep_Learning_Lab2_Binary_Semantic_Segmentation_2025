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


# 添加DICE損失函數
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # 應用sigmoid將logits轉換為概率
        pred = torch.sigmoid(pred)

        # 平滑處理
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / \
            (pred.sum() + target.sum() + self.smooth)

        return 1 - dice


def train(args):
    # 檢查 CUDA 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")

    # 建立儲存目錄
    os.makedirs('./saved_models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    # 設定TensorBoard
    log_dir = f'./logs/{time.strftime("%Y%m%d_%H%M%S")}_{args.model_type}'
    writer = SummaryWriter(log_dir)

    # 建立CSV記錄檔案
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = f'./logs/{timestamp}_{args.model_type}_training_log.csv'
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Epoch', 'Train_Loss', 'Val_Dice', 'Learning_Rate'])

    # 初始化記錄列表
    train_losses = []
    val_dices = []
    learning_rates = []

    # 加載數據集
    train_dataset = load_dataset(args.data_path, mode="train")
    val_dataset = load_dataset(args.data_path, mode="valid")

    # 創建數據加載器
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 初始化模型
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
            "不支持的模型類型。請選擇 'unet'、'unet_large'、'unet_xl' 或 'resnet34_unet'")

    model = model.to(device)

    # 定義損失函數和優化器
    criterion = DiceLoss(smooth=1.0)  # 使用DICE損失函數

    # 使用AdamW優化器，提供更好的權重衰減
    optimizer = optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # 使用OneCycleLR學習率調度器，提供更好的收斂性能
    total_steps = len(train_loader) * args.epochs
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.learning_rate,
                                              total_steps=total_steps,
                                              pct_start=0.3,  # 30%時間用於預熱
                                              div_factor=25,  # 初始學習率 = max_lr/div_factor
                                              final_div_factor=1000)  # 最終學習率 = max_lr/(div_factor*final_div_factor)

    # 訓練循環
    best_dice = 0.0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        batch_losses = []  # 記錄每個批次的損失

        # 訓練
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch') as pbar:
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(device, dtype=torch.float32)
                true_masks = batch['mask'].to(device, dtype=torch.float32)

                # 前向傳播
                outputs = model(images)
                loss = criterion(outputs, true_masks)

                # 反向傳播和優化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # 記錄當前批次損失
                batch_loss = loss.item()
                epoch_loss += batch_loss
                batch_losses.append(batch_loss)

                # 記錄到TensorBoard (每100個批次)
                if batch_idx % 100 == 0:
                    writer.add_scalar('Batch Loss', batch_loss,
                                      epoch * len(train_loader) + batch_idx)
                    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'],
                                      epoch * len(train_loader) + batch_idx)

                pbar.update(1)
                pbar.set_postfix(
                    {'loss': batch_loss, 'lr': optimizer.param_groups[0]['lr']})

        # 計算平均訓練損失
        avg_train_loss = epoch_loss / len(train_loader)

        # 驗證
        val_dice = evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]['lr']

        # 記錄到列表
        train_losses.append(avg_train_loss)
        val_dices.append(val_dice)
        learning_rates.append(current_lr)

        # 記錄到TensorBoard
        writer.add_scalar('Training Loss/epoch', avg_train_loss, epoch)
        writer.add_scalar('Validation Dice/epoch', val_dice, epoch)
        writer.add_scalar('Learning Rate/epoch', current_lr, epoch)

        # 記錄到CSV
        csv_writer.writerow([epoch+1, avg_train_loss, val_dice, current_lr])
        csv_file.flush()  # 立即寫入，不等待緩衝

        print(
            f'Epoch {epoch+1}/{args.epochs}, Loss: {avg_train_loss:.4f}, 驗證 Dice: {val_dice:.4f}, LR: {current_lr:.6f}')

        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            model_path = f'./saved_models/{time.strftime("%Y%m%d_%H%M%S")}_{args.model_type}_best_model_{best_dice:.4f}.pth'
            torch.save(model.state_dict(), model_path)
            print(f'保存新的最佳模型，Dice: {best_dice:.4f}，路徑: {model_path}')

    # 保存最終模型
    final_model_path = f'./saved_models/{args.model_type}_final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f'訓練完成。最佳驗證 Dice: {best_dice:.4f}，最終模型路徑: {final_model_path}')

    # 關閉CSV文件和TensorBoard
    csv_file.close()
    writer.close()

    # 繪製並保存學習曲線
    plot_learning_curves(train_losses, val_dices,
                         learning_rates, args.model_type)

    return {
        'train_losses': train_losses,
        'val_dices': val_dices,
        'learning_rates': learning_rates,
        'best_dice': best_dice
    }


def plot_learning_curves(train_losses, val_dices, learning_rates, model_type):
    """繪製學習曲線並保存為圖片"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # 創建圖表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # 繪製訓練損失和驗證Dice
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'bo-', label='訓練損失')
    ax1.plot(epochs, val_dices, 'ro-', label='驗證Dice係數')
    ax1.set_title(f'{model_type} 模型訓練曲線')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('損失 / Dice係數')
    ax1.legend()
    ax1.grid(True)

    # 繪製學習率
    ax2.plot(epochs, learning_rates, 'go-', label='學習率')
    ax2.set_title('學習率變化')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('學習率')
    ax2.legend()
    ax2.grid(True)

    # 調整佈局並保存
    plt.tight_layout()
    plt.savefig(f'./logs/{timestamp}_{model_type}_learning_curves.png')
    plt.close()
    print(f'學習曲線已保存到: ./logs/{timestamp}_{model_type}_learning_curves.png')


def get_args():
    parser = argparse.ArgumentParser(
        description='使用圖像和目標遮罩訓練 UNet')
    parser.add_argument('--data_path', type=str,
                        help='輸入數據的路徑', default='./dataset/oxford-iiit-pet')
    parser.add_argument('--epochs', '-e', type=int,
                        default=10, help='訓練周期數')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=16, help='批次大小')
    parser.add_argument('--learning-rate', '-lr', type=float,
                        default=3e-4, help='學習率')
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'unet_large',
                                 'unet_xl', 'resnet34_unet'],
                        help='模型架構 (unet, unet_large, unet_xl 或 resnet34_unet)')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
