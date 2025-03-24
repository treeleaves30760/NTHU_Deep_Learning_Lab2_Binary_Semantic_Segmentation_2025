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

        # 訓練
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{args.epochs}', unit='batch') as pbar:
            for batch in train_loader:
                images = batch['image'].to(device, dtype=torch.float32)
                true_masks = batch['mask'].to(device, dtype=torch.float32)

                # 前向傳播
                outputs = model(images)
                loss = criterion(outputs, true_masks)

                # 反向傳播和優化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()  # 每個batch更新學習率

                epoch_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(
                    {'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})

        # 驗證
        val_dice = evaluate(model, val_loader, device)

        print(
            f'Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_loader):.4f}, 驗證 Dice: {val_dice:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(),
                       f'./saved_models/{time.strftime("%Y%m%d_%H%M%S")}_{args.model_type}_best_model.pth')
            print(f'保存新的最佳模型，Dice: {best_dice:.4f}')

    # 保存最終模型
    torch.save(model.state_dict(),
               f'./saved_models/{time.strftime("%Y%m%d_%H%M%S")}_{args.model_type}_final_model.pth')
    print(f'訓練完成。最佳驗證 Dice: {best_dice:.4f}')


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
                        default=3e-4, help='學習率')  # 修改默認學習率為3e-4
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'unet_large',
                                 'unet_xl', 'resnet34_unet'],
                        help='模型架構 (unet, unet_large, unet_xl 或 resnet34_unet)')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
