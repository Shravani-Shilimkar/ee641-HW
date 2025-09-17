# problem2/evaluate.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from model import HeatmapNet, RegressionNet
from dataset import KeypointDataset
from torch.utils.data import DataLoader

def extract_keypoints_from_heatmaps(heatmaps):
    """
    Extracts (x, y) coordinates from heatmaps by finding the argmax.
    
    Args:
        heatmaps (Tensor): Heatmaps of shape [batch, num_keypoints, H, W].
        
    Returns:
        coords (Tensor): Coordinates of shape [batch, num_keypoints, 2] in [0, 1] range.
    """
    batch_size, num_keypoints, h, w = heatmaps.shape
    
    # Flatten the height and width dimensions
    heatmaps_flat = heatmaps.view(batch_size, num_keypoints, -1)
    
    # Find the index of the max value
    max_indices = torch.argmax(heatmaps_flat, dim=2)
    
    # Convert flat indices to 2D coordinates
    y_coords = (max_indices / w).float()
    x_coords = (max_indices % w).float()
    
    # Normalize coordinates to [0, 1]
    y_coords /= h
    x_coords /= w
    
    coords = torch.stack([x_coords, y_coords], dim=2)
    return coords

def compute_pck(predictions, ground_truths, thresholds):
    """
    Computes Percentage of Correct Keypoints (PCK).
    A keypoint is correct if the distance between the prediction and ground truth is
    within a certain threshold, normalized by the bounding box size of the ground truth.
    
    Args:
        predictions (Tensor): Predicted keypoints of shape [N, num_keypoints, 2].
        ground_truths (Tensor): Ground truth keypoints of shape [N, num_keypoints, 2].
        thresholds (list): List of threshold values.
        
    Returns:
        pck_values (dict): Dictionary mapping threshold to accuracy.
    """
    pck_values = {}
    
    for t in thresholds:
        correct_count = 0
        total_count = 0
        
        for i in range(predictions.shape[0]):
            pred_kps = predictions[i] # [num_keypoints, 2]
            gt_kps = ground_truths[i]   # [num_keypoints, 2]
            
            # Bbox normalization factor
            gt_min = gt_kps.min(axis=0).values
            gt_max = gt_kps.max(axis=0).values
            bbox_diag = torch.linalg.norm(gt_max - gt_min)
            if bbox_diag < 1e-6: continue # Avoid division by zero for single points

            # Calculate L2 distance
            distances = torch.linalg.norm(pred_kps - gt_kps, axis=1)
            
            # Check if distance is within threshold
            correct = (distances / bbox_diag) <= t
            correct_count += correct.sum().item()
            total_count += len(correct)
            
        pck_values[t] = correct_count / total_count if total_count > 0 else 0.0

    return pck_values


def plot_pck_curves(pck_heatmap, pck_regression, save_path):
    """Plots PCK curves for both methods."""
    thresholds = sorted(pck_heatmap.keys())
    heatmap_acc = [pck_heatmap[t] for t in thresholds]
    regression_acc = [pck_regression[t] for t in thresholds]

    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, heatmap_acc, 'o-', label='Heatmap Method')
    plt.plot(thresholds, regression_acc, 's-', label='Regression Method')
    plt.title('PCK Comparison')
    plt.xlabel('Normalized Distance Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    print(f"PCK curve saved to {save_path}")

def visualize_predictions(image, pred_keypoints, gt_keypoints, save_path):
    """Visualizes predictions and ground truths on an image."""
    image = image.squeeze(0).cpu().numpy()
    pred_keypoints = pred_keypoints.cpu().numpy() * image.shape[0] # Scale to image size
    gt_keypoints = gt_keypoints.cpu().numpy() * image.shape[0]

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.scatter(gt_keypoints[:, 0], gt_keypoints[:, 1], c='r', marker='x', label='Ground Truth')
    plt.scatter(pred_keypoints[:, 0], pred_keypoints[:, 1], c='g', marker='o', facecolors='none', label='Prediction')
    plt.title('Prediction vs. Ground Truth')
    plt.legend()
    plt.axis('off')
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    # --- Main Evaluation Logic ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs("results/visualizations", exist_ok=True)
    
    # Load test dataset
    test_dataset = KeypointDataset(
        image_dir='data/test/images', 
        annotation_file='data/test/annotations.json', 
        output_type='regression' # Use regression for GT coordinates
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load models
    heatmap_model = HeatmapNet().to(device)
    heatmap_model.load_state_dict(torch.load('results/heatmap_model.pth', map_location=device))
    heatmap_model.eval()

    regression_model = RegressionNet().to(device)
    regression_model.load_state_dict(torch.load('results/regression_model.pth', map_location=device))
    regression_model.eval()
    
    # Collect all predictions and ground truths
    all_hm_preds, all_reg_preds, all_gts = [], [], []
    with torch.no_grad():
        for images, gt_coords in tqdm(test_loader, desc="Evaluating models"):
            images = images.to(device)
            num_keypoints = gt_coords.shape[1] // 2
            
            # Ground truths
            gt_kps = gt_coords.view(-1, num_keypoints, 2)
            all_gts.append(gt_kps)
            
            # Heatmap predictions
            heatmaps = heatmap_model(images)
            hm_preds = extract_keypoints_from_heatmaps(heatmaps.cpu())
            all_hm_preds.append(hm_preds)

            # Regression predictions
            reg_coords = regression_model(images).cpu()
            reg_preds = reg_coords.view(-1, num_keypoints, 2)
            all_reg_preds.append(reg_preds)

    all_hm_preds = torch.cat(all_hm_preds, dim=0)
    all_reg_preds = torch.cat(all_reg_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)

    # Compute PCK
    thresholds = [0.05, 0.1, 0.15, 0.2]
    pck_heatmap = compute_pck(all_hm_preds, all_gts, thresholds)
    pck_regression = compute_pck(all_reg_preds, all_gts, thresholds)
    
    print("PCK Results (Heatmap):", pck_heatmap)
    print("PCK Results (Regression):", pck_regression)

    # Plot PCK curves
    plot_pck_curves(pck_heatmap, pck_regression, 'results/visualizations/pck_curve.png')

    # Visualize some sample predictions
    for i in range(5):
        img, _ = test_dataset[i]
        visualize_predictions(img, all_hm_preds[i], all_gts[i], f'results/visualizations/sample_{i}_heatmap.png')
        visualize_predictions(img, all_reg_preds[i], all_gts[i], f'results/visualizations/sample_{i}_regression.png')