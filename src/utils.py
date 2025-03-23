def dice_score(pred_mask, gt_mask):
    """
    計算預測遮罩和真實遮罩之間的 Dice 分數

    Args:
        pred_mask: 預測的二值遮罩
        gt_mask: 真實的二值遮罩

    Returns:
        float: Dice 分數
    """
    smooth = 1e-7  # 避免除以零

    # 將遮罩攤平
    pred_mask = pred_mask.view(-1)
    gt_mask = gt_mask.view(-1)

    # 計算交集和並集
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()

    # 計算 Dice 分數
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice
