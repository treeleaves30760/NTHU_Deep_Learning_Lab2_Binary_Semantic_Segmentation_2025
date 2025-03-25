def dice_score(pred_mask, gt_mask):
    """
    Calculate the Dice score between predicted mask and ground truth mask

    Args:
        pred_mask: Predicted binary mask
        gt_mask: Ground truth binary mask

    Returns:
        float: Dice score
    """
    smooth = 1e-7  # Avoid division by zero

    # Flatten masks
    pred_mask = pred_mask.view(-1)
    gt_mask = gt_mask.view(-1)

    # Calculate intersection and union
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum()

    # Calculate Dice score
    dice = (2. * intersection + smooth) / (union + smooth)

    return dice
