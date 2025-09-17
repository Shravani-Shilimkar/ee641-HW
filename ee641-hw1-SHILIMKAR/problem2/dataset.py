# problem2/dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import os

class KeypointDataset(Dataset):
    """
    Dataset for keypoint detection.
    
    Handles data loading and target generation for both heatmap and regression approaches.
    """
    def __init__(self, image_dir, annotation_file, output_type='heatmap', 
                 image_size=128, heatmap_size=64, sigma=2.0):
        """
        Args:
            image_dir (str): Path to the directory containing images.
            annotation_file (str): Path to the JSON annotation file.
            output_type (str): 'heatmap' or 'regression'.
            image_size (int): The size to which images will be resized.
            heatmap_size (int): The size of the output heatmaps.
            sigma (float): The standard deviation of the Gaussian kernel for heatmaps.
        """
        self.image_dir = image_dir
        self.output_type = output_type
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

        if output_type not in ['heatmap', 'regression']:
            raise ValueError("output_type must be 'heatmap' or 'regression'")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Returns a sample from the dataset.
        
        Returns:
            image (Tensor): Image tensor of shape [1, image_size, image_size].
            target (Tensor): Target tensor.
                - If output_type is 'heatmap', shape is [num_keypoints, heatmap_size, heatmap_size].
                - If output_type is 'regression', shape is [num_keypoints * 2].
        """
        annotation = self.annotations[idx]
        image_path = os.path.join(self.image_dir, annotation['image_path'])
        
        # Load and process image
        image = Image.open(image_path).convert('L') # Grayscale
        original_size = image.size
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) # [1, H, W]

        # Load and scale keypoints
        keypoints = np.array(annotation['keypoints'])
        
        if self.output_type == 'heatmap':
            # Scale keypoints to heatmap size
            scale_x = self.heatmap_size / original_size[0]
            scale_y = self.heatmap_size / original_size[1]
            scaled_keypoints = keypoints * np.array([scale_x, scale_y])
            target = self.generate_heatmaps(scaled_keypoints, self.heatmap_size, self.heatmap_size)
        else: # regression
            # Normalize keypoints to [0, 1]
            normalized_keypoints = keypoints / np.array(original_size)
            target = torch.from_numpy(normalized_keypoints.flatten()).float()
            
        return image_tensor, target

    def generate_heatmaps(self, keypoints, height, width):
        """
        Generates Gaussian heatmaps for keypoints.
        """
        num_keypoints = len(keypoints)
        heatmaps = np.zeros((num_keypoints, height, width), dtype=np.float32)
        
        for i in range(num_keypoints):
            x, y = keypoints[i]
            
            # Create a grid of coordinates
            xx, yy = np.meshgrid(np.arange(width), np.arange(height))
            
            # Generate Gaussian heatmap
            heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))
            
            # Handle boundary cases: clamp values
            heatmap[heatmap < np.finfo(heatmap.dtype).eps * heatmap.max()] = 0
            heatmaps[i] = heatmap

        return torch.from_numpy(heatmaps)