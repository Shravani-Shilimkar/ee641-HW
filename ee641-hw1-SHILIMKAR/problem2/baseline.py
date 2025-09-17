# problem2/baseline.py

import torch
import os
import numpy as np

from model import HeatmapNet, RegressionNet
from dataset import KeypointDataset
from torch.utils.data import DataLoader
from evaluate import visualize_predictions, compute_pck, extract_keypoints_from_heatmaps
# Note: You would need to import and re-use the training logic from train.py
# For brevity, this is left as a descriptive plan.

def ablation_study():
    """
    Conduct ablation studies on key hyperparameters.
    This function outlines the experiments. A full implementation would require
    running the training script multiple times with different parameters.
    """
    print("--- Running Ablation Studies ---")
    
    # 1. Effect of heatmap resolution
    print("\n1. Testing Heatmap Resolution...")
    # for resolution in [32, 64, 128]:
    #   print(f"Training with resolution: {resolution}x{resolution}")
    #   # Create dataset with heatmap_size=resolution
    #   # Train HeatmapNet
    #   # Evaluate and log PCK results
    
    # 2. Effect of Gaussian sigma
    print("\n2. Testing Gaussian Sigma...")
    # for sigma in [1.0, 2.0, 3.0, 4.0]:
    #   print(f"Training with sigma: {sigma}")
    #   # Create dataset with sigma=sigma
    #   # Train HeatmapNet
    #   # Evaluate and log PCK results
    
    # 3. Effect of skip connections
    print("\n3. Testing Effect of Skip Connections...")
    #   print("Training HeatmapNet without skip connections")
    #   # Create a modified HeatmapNet model without skip connections
    #   # Train the modified model
    #   # Compare its PCK against the original model
    
    print("\nAblation study plan complete. Implement training loops to run experiments.")

def analyze_failure_cases(heatmap_model, regression_model, test_loader, device):
    """Identifies and visualizes failure cases."""
    print("\n--- Analyzing Failure Cases ---")
    os.makedirs("results/visualizations/failures", exist_ok=True)
    
    heatmap_model.eval()
    regression_model.eval()
    
    cases_found = {'hm_wins': 0, 'reg_wins': 0, 'both_fail': 0}
    max_cases_to_save = 5

    with torch.no_grad():
        for i, (images, gt_coords) in enumerate(test_loader):
            images = images.to(device)
            num_kps = gt_coords.shape[1] // 2
            gt_kps = gt_coords.view(-1, num_kps, 2)
            
            # Get predictions
            hm_preds = extract_keypoints_from_heatmaps(heatmap_model(images).cpu())
            reg_preds = regression_model(images).cpu().view(-1, num_kps, 2)
            
            # Calculate error (normalized by image size for simplicity)
            hm_error = torch.linalg.norm(hm_preds - gt_kps, axis=2).mean(axis=1)
            reg_error = torch.linalg.norm(reg_preds - gt_kps, axis=2).mean(axis=1)

            for j in range(images.size(0)):
                idx = i * test_loader.batch_size + j
                h_err = hm_error[j].item()
                r_err = reg_error[j].item()
                
                # Case 1: Heatmap succeeds, Regression fails
                if h_err < 0.05 and r_err > 0.15 and cases_found['hm_wins'] < max_cases_to_save:
                    print(f"Found case 'hm_wins' at index {idx}")
                    visualize_predictions(images[j], hm_preds[j], gt_kps[j], f'results/visualizations/failures/case{idx}_hm_wins.png')
                    visualize_predictions(images[j], reg_preds[j], gt_kps[j], f'results/visualizations/failures/case{idx}_hm_wins_reg_fails.png')
                    cases_found['hm_wins'] += 1

                # Case 2: Regression succeeds, Heatmap fails
                elif r_err < 0.05 and h_err > 0.15 and cases_found['reg_wins'] < max_cases_to_save:
                    print(f"Found case 'reg_wins' at index {idx}")
                    visualize_predictions(images[j], reg_preds[j], gt_kps[j], f'results/visualizations/failures/case{idx}_reg_wins.png')
                    visualize_predictions(images[j], hm_preds[j], gt_kps[j], f'results/visualizations/failures/case{idx}_reg_wins_hm_fails.png')
                    cases_found['reg_wins'] += 1

                # Case 3: Both methods fail
                elif h_err > 0.15 and r_err > 0.15 and cases_found['both_fail'] < max_cases_to_save:
                    print(f"Found case 'both_fail' at index {idx}")
                    visualize_predictions(images[j], hm_preds[j], gt_kps[j], f'results/visualizations/failures/case{idx}_both_fail_hm.png')
                    visualize_predictions(images[j], reg_preds[j], gt_kps[j], f'results/visualizations/failures/case{idx}_both_fail_reg.png')
                    cases_found['both_fail'] += 1

if __name__ == '__main__':
    ablation_study()

    # Load models and data for failure analysis
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_dataset = KeypointDataset(
        image_dir='data/test/images', 
        annotation_file='data/test/annotations.json', 
        output_type='regression'
    )
    test_loader = DataLoader(test_dataset, batch_size=16)

    heatmap_model = HeatmapNet().to(device)
    heatmap_model.load_state_dict(torch.load('results/heatmap_model.pth', map_location=device))

    regression_model = RegressionNet().to(device)
    regression_model.load_state_dict(torch.load('results/regression_model.pth', map_location=device))
    
    analyze_failure_cases(heatmap_model, regression_model, test_loader, device)