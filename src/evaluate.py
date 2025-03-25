import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import dice_score


def evaluate(net, data, device):
    """
    Evaluate the network performance on a given dataset

    Args:
        net: Neural network model
        data: Data loader
        device: torch.device

    Returns:
        float: Average Dice score
    """
    net.eval()
    num_val_batches = len(data)
    dice_scores = 0

    # Disable gradient calculation to speed up inference
    with torch.no_grad():
        # Iterate through validation set
        with tqdm(total=num_val_batches, desc='Validating', unit='batch', leave=False) as pbar:
            for batch in data:
                images = batch['image'].to(device, dtype=torch.float32)
                true_masks = batch['mask'].to(device, dtype=torch.float32)

                # Prediction
                mask_pred = net(images)

                # Apply sigmoid activation
                mask_pred = torch.sigmoid(mask_pred)

                # Convert to binary mask (0 or 1)
                mask_pred = (mask_pred > 0.5).float()

                # Calculate Dice score
                dice_scores += dice_score(mask_pred, true_masks)

                pbar.update(1)

    # Calculate average Dice score
    avg_dice = dice_scores / num_val_batches

    # Switch back to training mode
    net.train()

    return avg_dice
