import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import dice_score


def evaluate(net, data, device):
    """
    評估網絡在給定數據集上的表現

    Args:
        net: 神經網絡模型
        data: 數據加載器
        device: torch.device

    Returns:
        float: 平均 Dice 分數
    """
    net.eval()
    num_val_batches = len(data)
    dice_scores = 0

    # 禁用梯度計算以加速推理
    with torch.no_grad():
        # 迭代驗證集
        with tqdm(total=num_val_batches, desc='驗證中', unit='batch', leave=False) as pbar:
            for batch in data:
                images = batch['image'].to(device, dtype=torch.float32)
                true_masks = batch['mask'].to(device, dtype=torch.float32)

                # 預測
                mask_pred = net(images)

                # 應用 sigmoid 激活
                mask_pred = torch.sigmoid(mask_pred)

                # 轉換為二值遮罩 (0 或 1)
                mask_pred = (mask_pred > 0.5).float()

                # 計算 Dice 分數
                dice_scores += dice_score(mask_pred, true_masks)

                pbar.update(1)

    # 計算平均 Dice 分數
    avg_dice = dice_scores / num_val_batches

    # 切換回訓練模式
    net.train()

    return avg_dice
