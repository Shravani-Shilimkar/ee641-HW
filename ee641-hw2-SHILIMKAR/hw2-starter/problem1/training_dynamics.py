# """
# GAN training implementation with mode collapse analysis.
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# from collections import defaultdict

# def train_gan(generator, discriminator, data_loader, num_epochs=100, device='cuda'):
#     """
#     Standard GAN training implementation.
    
#     Uses vanilla GAN objective which typically exhibits mode collapse.
    
#     Args:
#         generator: Generator network
#         discriminator: Discriminator network
#         data_loader: DataLoader for training data
#         num_epochs: Number of training epochs
#         device: Device for computation
        
#     Returns:
#         dict: Training history and metrics
#     """
#     # Initialize optimizers
#     g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#     d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
#     # Loss function
#     criterion = nn.BCELoss()
    
#     # Training history
#     history = defaultdict(list)
    
#     for epoch in range(num_epochs):
#         for batch_idx, (real_images, labels) in enumerate(data_loader):
#             batch_size = real_images.size(0)
#             real_images = real_images.to(device)
            
#             # Labels for loss computation
#             real_labels = torch.ones(batch_size, 1).to(device)
#             fake_labels = torch.zeros(batch_size, 1).to(device)
            
#             # ========== Train Discriminator ==========
#             # TODO: Implement discriminator training step
#             # 1. Zero gradients
#             # 2. Forward pass on real images
#             # 3. Compute real loss
#             # 4. Generate fake images from random z
#             # 5. Forward pass on fake images (detached)
#             # 6. Compute fake loss
#             # 7. Backward and optimize
#             d_optimizer.zero_grad()
            
#             # 1. On real images
#             d_output_real = discriminator(real_images)
#             d_loss_real = criterion(d_output_real, real_labels)
            
#             # 2. On fake images
#             z = torch.randn(batch_size, z_dim).to(device)
#             fake_images = generator(z)
#             d_output_fake = discriminator(fake_images.detach())
#             d_loss_fake = criterion(d_output_fake, fake_labels)
            
#             # 3. Total loss and optimization
#             d_loss = d_loss_real + d_loss_fake
#             d_loss.backward()
#             d_optimizer.step()
            
#             # ========== Train Generator ==========
#             # TODO: Implement generator training step
#             # 1. Zero gradients
#             # 2. Generate fake images
#             # 3. Forward pass through discriminator
#             # 4. Compute adversarial loss
#             # 5. Backward and optimize
#             g_optimizer.zero_grad()
            
#             # We need to run the fake images through the discriminator again
#             # as the discriminator has been updated.
#             d_output_for_g = discriminator(fake_images)
#             g_loss = criterion(d_output_for_g, real_labels)
            
#             g_loss.backward()
#             g_optimizer.step()
            
#             # Log metrics
#             if batch_idx % 10 == 0:
#                 history['d_loss'].append(d_loss.item())
#                 history['g_loss'].append(g_loss.item())
#                 history['epoch'].append(epoch + batch_idx/len(data_loader))
        
#         # Analyze mode collapse every 10 epochs
#         if epoch % 10 == 0:
#             mode_coverage = analyze_mode_coverage(generator, device)
#             history['mode_coverage'].append(mode_coverage)
#             print(f"Epoch {epoch}: Mode coverage = {mode_coverage:.2f}")
    
#     return history

# def analyze_mode_coverage(generator, device, z_dim=100, n_samples=1000):
#     """
#     Measure mode coverage by counting unique letters in generated samples.
    
#     Args:
#         generator: Trained generator network
#         device: Device for computation
#         n_samples: Number of samples to generate
        
#     Returns:
#         float: Coverage score (unique letters / 26)
#     """
#     # TODO: Generate n_samples images
#     # Use provided letter classifier to identify generated letters
#     # Count unique letters produced
#     # Return coverage score (0 to 1)
#     generator.eval() # Set generator to evaluation mode
#     classifier = get_letter_classifier(device) # Load a pre-trained classifier
    
#     with torch.no_grad():
#         z = torch.randn(n_samples, z_dim).to(device)
#         generated_images = generator(z)
#         # Rescale from [-1, 1] to [0, 1] if your classifier expects that
#         generated_images = (generated_images + 1) / 2
        
#         predictions = classifier(generated_images)
#         predicted_labels = torch.argmax(predictions, dim=1)
        
#         unique_labels = torch.unique(predicted_labels).cpu().numpy()
    
#     generator.train() # Set back to train mode
    
#     coverage_score = len(unique_labels) / 26.0
#     letter_counts = {chr(65 + i): int((predicted_labels == i).sum()) for i in range(26)}
    
#     return coverage_score, letter_counts
    

# def visualize_mode_collapse(history, save_path):
#     """
#     Visualize mode collapse progression over training.
    
#     Args:
#         history: Training metrics dictionary
#         save_path: Output path for visualization
#     """
#     # TODO: Plot mode coverage over time
#     # Show which letters survive and which disappear
#     epochs = [item['epoch'] for item in history['mode_coverage']]
#     coverage_scores = [item['coverage'] * 26 for item in history['mode_coverage']]

#     plt.figure(figsize=(10, 5))
#     plt.plot(epochs, coverage_scores, marker='o', linestyle='-')
#     plt.title('Mode Coverage Over Training')
#     plt.xlabel('Epoch')
#     plt.ylabel('Number of Unique Letters Generated (out of 26)')
#     plt.ylim(0, 27)
#     plt.grid(True)
#     plt.savefig(save_path)
#     print(f"Mode collapse visualization saved to {save_path}")
    


"""
GAN training implementation with mode collapse analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import os

# NOTE: For analyze_mode_coverage to work, you must provide a
# trained classifier for the letters. This is a placeholder.
def get_letter_classifier(device):
    model = nn.Sequential(nn.Flatten(), nn.Linear(28*28, 26)).to(device)
    # e.g., model.load_state_dict(torch.load('path/to/classifier.pth'))
    model.eval()
    return model

def train_gan(generator, discriminator, data_loader, num_epochs=100, device='cuda'):
    """
    Standard GAN training implementation.
    """
    z_dim = generator.z_dim
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    history = defaultdict(list)
    
    fixed_z = torch.randn(64, z_dim, device=device)
    os.makedirs("results/visualizations", exist_ok=True)
    
    for epoch in range(num_epochs):
        for batch_idx, (real_images, labels) in enumerate(data_loader):
            batch_size = real_images.size(0)
            real_images = (real_images * 2 - 1).to(device)
            
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ========== Train Discriminator ==========
            d_optimizer.zero_grad()
            d_output_real = discriminator(real_images)
            d_loss_real = criterion(d_output_real, real_labels)
            z = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(z)
            d_output_fake = discriminator(fake_images.detach())
            d_loss_fake = criterion(d_output_fake, fake_labels)
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # ========== Train Generator ==========
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(z)
            g_output = discriminator(fake_images)
            g_loss = criterion(g_output, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{len(data_loader)}], D_Loss: {d_loss.item():.4f}, G_Loss: {g_loss.item():.4f}")
                history['d_loss'].append(d_loss.item())
                history['g_loss'].append(g_loss.item())
        
        # Save images at specified epochs
        if epoch + 1 in [10, 30, 50, 100]:
            generator.eval()
            with torch.no_grad():
                fake_images_out = generator(fixed_z).detach().cpu()
                save_path = f"results/visualizations/generated_epoch_{epoch+1}.png"
                save_image(fake_images_out, save_path, nrow=8, normalize=True)
            generator.train()

        # Analyze mode collapse
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            coverage, _ = analyze_mode_coverage(generator, device)
            history['mode_coverage'].append({'epoch': epoch + 1, 'coverage': coverage})
            print(f"Epoch {epoch+1}: Mode coverage = {coverage:.2f}")
    
    return history

def analyze_mode_coverage(generator, device, n_samples=1000):
    generator.eval()
    classifier = get_letter_classifier(device)
    z_dim = generator.z_dim
    
    with torch.no_grad():
        z = torch.randn(n_samples, z_dim, device=device)
        generated_images = generator(z)
        generated_images = (generated_images + 1) / 2.0
        
        predictions = classifier(generated_images)
        predicted_labels = torch.argmax(predictions, dim=1)
        
        unique_labels_found = torch.unique(predicted_labels)
        coverage = len(unique_labels_found) / 26.0
        
        letter_counts = {chr(65+i): int((predicted_labels == i).sum()) for i in range(26)}
        
    generator.train()
    return coverage, letter_counts

def visualize_mode_collapse(history, save_path):
    epochs = [item['epoch'] for item in history['mode_coverage']]
    coverage_scores = [item['coverage'] for item in history['mode_coverage']]

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, coverage_scores, marker='o', linestyle='-')
    plt.title('Mode Coverage Over Training Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mode Coverage (Unique Letters / 26)')
    plt.grid(True)
    plt.ylim(0, 1.05)
    plt.savefig(save_path)
    plt.close()