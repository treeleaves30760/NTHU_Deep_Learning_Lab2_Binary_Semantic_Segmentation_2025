import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.unet import UNet, UNetLarge, UNetXL
from models.resnet34_unet import ResNet34_UNet
from oxford_pet import load_dataset
from utils import dice_score


def predict_img(net, image, device):
    """
    Predict segmentation result for an image

    Args:
        net: Model
        image: Input image tensor
        device: torch.device

    Returns:
        tensor: Predicted binary mask
    """
    net.eval()

    with torch.no_grad():
        # Move image to device
        image = image.to(device=device, dtype=torch.float32)

        # Prediction
        output = net(image)
        probs = torch.sigmoid(output)

        # Convert to binary mask (0 or 1)
        pred_mask = (probs > 0.5).float()

    return pred_mask


def get_args():
    parser = argparse.ArgumentParser(
        description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth',
                        help='Path to stored model weights')
    parser.add_argument('--data_path', type=str,
                        help='Path to input data', default='./dataset/oxford-iiit-pet')
    parser.add_argument('--batch_size', '-b', type=int,
                        default=1, help='Batch size')
    parser.add_argument('--model_type', type=str, default='unet',
                        choices=['unet', 'unet_large',
                                 'unet_xl', 'resnet34_unet'],
                        help='Model architecture (unet, unet_large, unet_xl or resnet34_unet)')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save output predictions')
    parser.add_argument('--save_results', type=bool, default=True,
                        help='Whether to save prediction results')
    parser.add_argument("-n", "--num_samples", type=int, default=100,
                        help='Number of samples to predict')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Check if output directory exists, create if it doesn't
    results_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(results_dir, exist_ok=True)

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    if args.model_type.lower() == 'unet':
        model = UNet(n_channels=3, n_classes=1, bilinear=True)
    elif args.model_type.lower() == 'unet_large':
        model = UNetLarge(n_channels=3, n_classes=1, bilinear=True)
    elif args.model_type.lower() == 'unet_xl':
        model = UNetXL(n_channels=3, n_classes=1, bilinear=True)
    elif args.model_type.lower() == 'resnet34_unet':
        model = ResNet34_UNet(n_classes=1, pretrained=False, bilinear=True)
    else:
        raise ValueError(
            "Unsupported model type. Please choose 'unet', 'unet_large', 'unet_xl' or 'resnet34_unet'")

    # Load model weights
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)

    # Load test dataset
    test_dataset = load_dataset(args.data_path, mode="test")
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    dice_scores_array = []
    count = 0
    # Predict and save results
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = batch['image']
        true_masks = batch['mask']

        # Predict masks
        pred_masks = predict_img(model, images, device)
        pred_masks = pred_masks.cpu()

        # Save results
        for j in range(images.shape[0]):
            if count >= args.num_samples:
                break

            dice_scores = dice_score(pred_masks[j], true_masks[j])
            dice_scores_array.append(dice_scores)

            # Convert tensors to numpy for visualization
            img = images[j].numpy().transpose(1, 2, 0)
            true_mask = true_masks[j].squeeze().numpy()
            pred_mask = pred_masks[j].squeeze().cpu().numpy()

            # Create figure with original image, ground truth mask and predicted mask, and display DICE Score in the Title
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(img)
            ax[0].set_title('Original Image')
            ax[0].axis('off')

            ax[1].imshow(true_mask, cmap='gray')
            ax[1].set_title('Ground Truth Mask')
            ax[1].axis('off')

            ax[2].imshow(pred_mask, cmap='gray')
            ax[2].set_title('Predicted Mask')
            ax[2].axis('off')

            # Display DICE Score in the Title
            fig.suptitle(f'DICE Score: {dice_scores:.4f}')

            # Save figure
            if args.save_results:
                plt.savefig(os.path.join(results_dir,
                            f'prediction_{i * args.batch_size + j}.png'))
                plt.close(fig)

            count += 1

        if count >= args.num_samples:
            break

    print(f"Prediction results saved to {results_dir}")
    print(f"DICE Score: {np.mean(dice_scores_array):.4f}")
