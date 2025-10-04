# """
# Latent space analysis tools for hierarchical VAE.
# """

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE

# def visualize_latent_hierarchy(model, data_loader, device='cuda'):
#     """
#     Visualize the two-level latent space structure.
    
#     TODO:
#     1. Encode all data to get z_high and z_low
#     2. Use t-SNE to visualize z_high (colored by genre)
#     3. For each z_high cluster, show z_low variations
#     4. Create hierarchical visualization
#     """
#     pass

# def interpolate_styles(model, pattern1, pattern2, n_steps=10, device='cuda'):
#     """
#     Interpolate between two drum patterns at both latent levels.
    
#     TODO:
#     1. Encode both patterns to get latents
#     2. Interpolate z_high (style transition)
#     3. Interpolate z_low (variation transition)
#     4. Decode and visualize both paths
#     5. Compare smooth vs abrupt transitions
#     """
#     pass

# def measure_disentanglement(model, data_loader, device='cuda'):
#     """
#     Measure how well the hierarchy disentangles style from variation.
    
#     TODO:
#     1. Group patterns by genre
#     2. Compute z_high variance within vs across genres
#     3. Compute z_low variance for same genre
#     4. Return disentanglement metrics
#     """
#     pass

# def controllable_generation(model, genre_labels, device='cuda'):
#     """
#     Test controllable generation using the hierarchy.
    
#     TODO:
#     1. Learn genre embeddings in z_high space
#     2. Generate patterns with specified genre
#     3. Control complexity via z_low sampling temperature
#     4. Evaluate genre classification accuracy
#     """
#     pass




import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from dataset import DrumPatternDataset
from hierarchical_vae import HierarchicalDrumVAE

def visualize_latent_hierarchy(model, data_loader, device='cuda'):
    model.eval()
    z_high_all = []
    z_low_all = []
    styles_all = []
    
    with torch.no_grad():
        for patterns, styles, _ in data_loader:
            patterns = patterns.to(device)
            _, _, z_low, _, _, z_high = model.encode_hierarchy(patterns)
            
            z_high_all.append(z_high.cpu().numpy())
            z_low_all.append(z_low.cpu().numpy())
            styles_all.append(styles.numpy())
            
    z_high_all = np.concatenate(z_high_all, axis=0)
    z_low_all = np.concatenate(z_low_all, axis=0)
    styles_all = np.concatenate(styles_all, axis=0)
    
    tsne = TSNE(n_components=2, random_state=42)
    z_high_tsne = tsne.fit_transform(z_high_all)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(z_high_tsne[:, 0], z_high_tsne[:, 1], c=styles_all, cmap='viridis', alpha=0.7)
    plt.legend(handles=scatter.legend_elements()[0], labels=data_loader.dataset.style_names)
    plt.title("t-SNE of High-Level Latent Space (z_high)")
    plt.show()

def interpolate_styles(model, pattern1, pattern2, n_steps=10, device='cuda'):
    model.eval()
    p1 = torch.from_numpy(pattern1).unsqueeze(0).to(device)
    p2 = torch.from_numpy(pattern2).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, _, z_low1, _, _, z_high1 = model.encode_hierarchy(p1)
        _, _, z_low2, _, _, z_high2 = model.encode_hierarchy(p2)
        
    interpolated_patterns = []
    for alpha in np.linspace(0, 1, n_steps):
        z_high_interp = (1 - alpha) * z_high1 + alpha * z_high2
        z_low_interp = (1 - alpha) * z_low1 + alpha * z_low2
        
        pattern_logits = model.decode_hierarchy(z_high_interp, z_low_interp)
        pattern = (torch.sigmoid(pattern_logits) > 0.5).float()
        interpolated_patterns.append(pattern.squeeze(0).cpu().numpy())
        
    return interpolated_patterns