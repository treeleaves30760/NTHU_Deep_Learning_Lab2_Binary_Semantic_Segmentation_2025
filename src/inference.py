import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset


def predict_img(net, image, device):
    """
    為圖像預測分割結果

    Args:
        net: 模型
        image: 輸入圖像張量
        device: torch.device

    Returns:
        tensor: 預測的二值遮罩
    """
    net.eval()

    with torch.no_grad():
        # 將圖像移至設備
        image = image.to(device=device, dtype=torch.float32)

        # 預測
        output = net(image)
        probs = torch.sigmoid(output)

        # 轉換為二值遮罩 (0 或 1)
        pred_mask = (probs > 0.5).float()

    return pred_mask


def get_args():
    parser = argparse.ArgumentParser(
        description='從輸入圖像預測遮罩')
    parser.add_argument('--model', default='MODEL.pth',
                        help='存儲的模型權重路徑')
    parser.add_argument('--data_path', type=str, help='輸入數據的路徑')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=1, help='批次大小')
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'resnet34_unet'],
                        help='模型架構 (unet 或 resnet34_unet)')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='保存輸出預測的目錄')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # 檢查輸出目錄是否存在，如果不存在則創建
    os.makedirs(args.output_dir, exist_ok=True)

    # 檢查 CUDA 是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加載模型
    if args.model_type.lower() == 'unet':
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
    elif args.model_type.lower() == 'resnet34_unet':
        model = ResNet34_UNet(n_classes=1, pretrained=False, bilinear=True)
    else:
        raise ValueError("不支持的模型類型。請選擇 'unet' 或 'resnet34_unet'")

    # 加載模型權重
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # 加載測試數據集
    test_dataset = load_dataset(args.data_path, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    # 預測並保存結果
    for i, batch in enumerate(test_loader):
        images = batch['image']
        true_masks = batch['mask']

        # 預測遮罩
        pred_masks = predict_img(model, images, device)

        # 保存結果
        for j in range(images.shape[0]):
            # 將張量轉換為 numpy 以進行可視化
            img = images[j].numpy().transpose(1, 2, 0)
            true_mask = true_masks[j].squeeze().numpy()
            pred_mask = pred_masks[j].squeeze().cpu().numpy()

            # 創建包含原始圖像、真實遮罩和預測遮罩的圖形
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(img)
            ax[0].set_title('原始圖像')
            ax[0].axis('off')

            ax[1].imshow(true_mask, cmap='gray')
            ax[1].set_title('真實遮罩')
            ax[1].axis('off')

            ax[2].imshow(pred_mask, cmap='gray')
            ax[2].set_title('預測遮罩')
            ax[2].axis('off')

            # 保存圖形
            plt.savefig(os.path.join(args.output_dir,
                        f'prediction_{i * args.batch_size + j}.png'))
            plt.close(fig)

    print(f"預測結果已保存至 {args.output_dir}")
