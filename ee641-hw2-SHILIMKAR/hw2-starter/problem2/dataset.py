"""
Dataset loader for drum pattern generation task.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os

class DrumPatternDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Initialize drum pattern dataset.
        
        Args:
            data_dir: Path to drum dataset directory
            split: 'train', 'val', or 'all'
        """
        self.data_dir = data_dir
        self.split = split
        
        # Load patterns from patterns.npz
        data_path = os.path.join(data_dir, 'patterns.npz')
        data = np.load(data_path)
        
        # Load the correct pre-split data based on the 'split' argument
        if split == 'train':
            self.patterns = data['train_patterns']
            self.styles = data['train_styles']
        elif split == 'val':
            self.patterns = data['val_patterns']
            self.styles = data['val_styles']
        elif split == 'all':
            # For analysis, combine the train and val sets
            self.patterns = np.concatenate((data['train_patterns'], data['val_patterns']), axis=0)
            self.styles = np.concatenate((data['train_styles'], data['val_styles']), axis=0)
        else:
            raise ValueError("Split must be 'train', 'val', or 'all'")
        
        # Load metadata from the correctly named 'patterns.json' file
        metadata_path = os.path.join(data_dir, 'patterns.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.instrument_names = metadata['instruments']
        self.style_names = metadata['styles']
    
    def __len__(self):
        return len(self.patterns)
    
    def __getitem__(self, idx):
        """
        Return a drum pattern sample.
        
        Returns:
            pattern: Binary tensor of shape [16, 9]
            style: Integer style label (0-4)
            density: Float indicating pattern density (for analysis)
        """
        pattern = self.patterns[idx]
        style = self.styles[idx]
        
        # Convert to tensor
        pattern_tensor = torch.from_numpy(pattern).float()
        
        # Compute density metric (fraction of active hits)
        density = pattern.sum() / (16 * 9)
        
        return pattern_tensor, style, density
    
    def pattern_to_pianoroll(self, pattern):
        """
        Convert pattern to visual piano roll representation.
        
        Args:
            pattern: Binary array [16, 9] or tensor
            
        Returns:
            pianoroll: Visual representation for plotting
        """
        if torch.is_tensor(pattern):
            pattern = pattern.cpu().numpy()
        
        # Create visual representation with instrument labels
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot each active hit
        for t in range(16):
            for i in range(9):
                if pattern[t, i] > 0.5:
                    rect = patches.Rectangle((t, i), 1, 1, 
                                            linewidth=1, 
                                            edgecolor='black',
                                            facecolor='blue')
                    ax.add_patch(rect)
        
        # Add grid
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_xticks(range(17))
        ax.set_yticks(range(10))
        ax.set_yticklabels([''] + self.instrument_names)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Instrument')
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        return fig