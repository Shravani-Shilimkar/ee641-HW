# """
# GAN stabilization techniques to combat mode collapse.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import copy
# import torch.optim as optim
# from collections import defaultdict
# from training_dynamics import analyze_mode_coverage # Import for analysis

# def feature_matching_loss(real_features, fake_features):
#     """
#     Calculate feature matching loss.
#     Match mean statistics: ||E[f(x)] - E[f(G(z))]||²
#     """
#     mean_real_features = torch.mean(real_features, 0)
#     mean_fake_features = torch.mean(fake_features, 0)
#     return torch.mean((mean_real_features - mean_fake_features) ** 2)


# def train_gan_with_fix(generator, discriminator, data_loader, 
#                        num_epochs=100, fix_type='feature_matching'):
#     """
#     Train GAN with mode collapse mitigation techniques.
    
#     Args:
#         generator: Generator network
#         discriminator: Discriminator network
#         data_loader: DataLoader for training data
#         num_epochs: Number of training epochs
#         fix_type: Stabilization method ('feature_matching', 'unrolled', 'minibatch')
        
#     Returns:
#         dict: Training history with metrics
#     """
    
#     if fix_type == 'feature_matching':
#         # Feature matching: Match statistics of intermediate layers
#         # instead of just final discriminator output
        
#         def feature_matching_loss(real_images, fake_images, discriminator):
#             """
#             TODO: Implement feature matching loss
            
#             Extract intermediate features from discriminator
#             Match mean statistics: ||E[f(x)] - E[f(G(z))]||²
#             Use discriminator.features (before final classifier)
#             """
#             pass
            
#     elif fix_type == 'unrolled':
#         # Unrolled GANs: Look ahead k discriminator updates
        
#         def unrolled_discriminator(discriminator, real_data, fake_data, k=5):
#             """
#             TODO: Implement k-step unrolled discriminator
            
#             Create temporary discriminator copy
#             Update it k times
#             Compute generator loss through updated discriminator
#             """
#             pass
            
#     elif fix_type == 'minibatch':
#         # Minibatch discrimination: Let discriminator see batch statistics
        
#         class MinibatchDiscrimination(nn.Module):
#             """
#             TODO: Add minibatch discrimination layer to discriminator
            
#             Compute L2 distance between samples in batch
#             Concatenate statistics to discriminator features
#             """
#             pass
    
#     # Training loop with chosen fix
#     # TODO: Implement modified training using selected technique
#     pass



"""
GAN stabilization techniques to combat mode collapse.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
import os # <-- Add os import
from torchvision.utils import save_image # <-- Add save_image import

from training_dynamics import analyze_mode_coverage


def train_gan_with_fix(generator, discriminator, data_loader, 
                       num_epochs=100, fix_type='feature_matching', device='cuda'):
    """
    Train GAN with mode collapse mitigation techniques.
    """
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_criterion = nn.BCELoss()
    history = defaultdict(list)
    z_dim = generator.z_dim

    # ADD THIS: Create a fixed noise vector for consistent visualization
    fixed_z = torch.randn(64, z_dim, device=device)
    os.makedirs("results/visualizations", exist_ok=True)

    if fix_type == 'feature_matching':
        def feature_matching_loss(real_images, fake_images, discriminator):
            real_features = discriminator.features(real_images)
            fake_features = discriminator.features(fake_images)
            mean_real_features = torch.mean(real_features, 0)
            mean_fake_features = torch.mean(fake_features, 0)
            return torch.mean((mean_real_features - mean_fake_features)**2)
        
        for epoch in range(num_epochs):
            for batch_idx, (real_images, _) in enumerate(data_loader):
                batch_size = real_images.size(0)
                real_images = (real_images * 2 - 1).to(device)

                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)
                
                # ... (Discriminator and Generator training steps remain the same) ...
                # Train Discriminator
                d_optimizer.zero_grad()
                d_output_real = discriminator(real_images)
                d_loss_real = d_criterion(d_output_real, real_labels)
                z = torch.randn(batch_size, z_dim, device=device)
                fake_images = generator(z)
                d_output_fake = discriminator(fake_images.detach())
                d_loss_fake = d_criterion(d_output_fake, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()

                # Train Generator with Feature Matching Loss
                g_optimizer.zero_grad()
                z = torch.randn(batch_size, z_dim, device=device)
                fake_images_for_g = generator(z)
                g_loss = feature_matching_loss(real_images, fake_images_for_g, discriminator)
                g_loss.backward()
                g_optimizer.step()
                
                if batch_idx % 100 == 0:
                    history['d_loss'].append(d_loss.item())
                    history['g_loss'].append(g_loss.item())

            # ADD THIS BLOCK to save images for the 'fixed' GAN
            if (epoch + 1) in [10, 30, 50, 100]:
                generator.eval()
                with torch.no_grad():
                    fake_images_out = generator(fixed_z).detach().cpu()
                    save_path = f"results/visualizations/generated_epoch_{epoch+1}_fixed.png"
                    save_image(fake_images_out, save_path, nrow=8, normalize=True)
                generator.train()

            # The mode coverage analysis block
            if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
                coverage, _ = analyze_mode_coverage(generator, device)
                history['mode_coverage'].append({'epoch': epoch + 1, 'coverage': coverage})
                print(f"Epoch {epoch+1} (Fixed): Mode coverage = {coverage:.2f}")

    else:
        raise NotImplementedError(f"Fix type '{fix_type}' is not implemented.")
        
    return history