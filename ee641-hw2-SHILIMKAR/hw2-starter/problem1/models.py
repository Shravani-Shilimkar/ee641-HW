# """
# GAN models for font generation.
# """

# import torch
# import torch.nn as nn

# class Generator(nn.Module):
#     def __init__(self, z_dim=100, conditional=False, num_classes=26):
#         """
#         Generator network that produces 28×28 letter images.
        
#         Args:
#             z_dim: Dimension of latent vector z
#             conditional: If True, condition on letter class
#             num_classes: Number of letter classes (26)
#         """
#         super().__init__()
#         self.z_dim = z_dim
#         self.conditional = conditional
        
#         # Calculate input dimension
#         input_dim = z_dim + (num_classes if conditional else 0)
        
#         # Architecture proven to work well for this task:
#         # Project and reshape: z → 7×7×128
#         self.project = nn.Sequential(
#             nn.Linear(input_dim, 128 * 7 * 7),
#             nn.BatchNorm1d(128 * 7 * 7),
#             nn.ReLU(True)
#         )
        
#         # Upsample: 7×7×128 → 14×14×64 → 28×28×1
#         self.main = nn.Sequential(
#             # TODO: Implement upsampling layers
#             # Use ConvTranspose2d with appropriate padding/stride
#             # Include BatchNorm2d and ReLU (except final layer)
#             # Final layer should use Tanh activation
#             # Input: [batch_size, 128, 7, 7]

#             nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
#             # Output: [batch_size, 64, 14, 14]
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
            
#             nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
#             # Output: [batch_size, 1, 28, 28]
#             nn.Tanh() # Tanh activation to scale output to [-1, 1]

#         )
    
#     def forward(self, z, class_label=None):
#         """
#         Generate images from latent code.
        
#         Args:
#             z: Latent vectors [batch_size, z_dim]
#             class_label: One-hot encoded class labels [batch_size, num_classes]
        
#         Returns:
#             Generated images [batch_size, 1, 28, 28] in range [-1, 1]
#         """
#         # TODO: Implement forward pass
#         # If conditional, concatenate z and class_label
#         # Project to spatial dimensions
#         # Apply upsampling network

#         if self.conditional:
#             # Ensure class_label is provided
#             if class_label is None:
#                 raise ValueError("Conditional generator requires class_label.")
#             # Concatenate noise vector z and class_label
#             z = torch.cat([z, class_label], 1)
                    
#         x = self.project(z)
#         x = x.view(-1, 128, 7, 7) # Reshape to [batch_size, 128, 7, 7]
#         output = self.main(x)
#         return output

        

# class Discriminator(nn.Module):
#     def __init__(self, conditional=False, num_classes=26):
#         """
#         Discriminator network that classifies 28×28 images as real/fake.
#         """
#         super().__init__()
#         self.conditional = conditional
        
#         # img_channels = 1 + (num_classes if conditional else 0)

#         # Proven architecture for 28×28 images:
#         self.features = nn.Sequential(
#             # TODO: Implement convolutional layers
#             # 28×28×1 → 14×14×64 → 7×7×128 → 3×3×256
#             # Use Conv2d with appropriate stride
#             # LeakyReLU(0.2) and Dropout2d(0.25)
#                         # Input: [batch_size, channels, 28, 28]
#             nn.Conv2d(img_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
#             # Output: [batch_size, 64, 14, 14]
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.25),
            
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
#             # Output: [batch_size, 128, 7, 7]
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.25),
            
#             nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0, bias=False),
#             # Output: [batch_size, 256, 3, 3]
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout2d(0.25)
#         )
        
#         # Calculate feature dimension after convolutions
#         feature_dim = 256 * 3 * 3  # Adjust based on your architecture
        
#         self.classifier = nn.Sequential(
#             nn.Linear(feature_dim + (num_classes if conditional else 0), 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, img, class_label=None):
#         """
#         Classify images as real (1) or fake (0).
        
#         Returns:
#             Probability of being real [batch_size, 1]
#         """
#         # TODO: Extract features, flatten, concatenate class if conditional

#         if self.conditional:
#             if class_label is None:
#                 raise ValueError("Conditional discriminator requires class_label.")
#             # Reshape label to be concatenated as a channel
#             label_plane = class_label.view(-1, self.num_classes, 1, 1)
#             label_plane = label_plane.repeat(1, 1, img.size(2), img.size(3))
#             img = torch.cat([img, label_plane], 1)

#         features = self.features(img)
#         features = features.view(features.size(0), -1) # Flatten
#         output = self.classifier(features)
#         return output
        


"""
GAN models for font generation.
"""

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, conditional=False, num_classes=26):
        """
        Generator network that produces 28×28 letter images.
        
        Args:
            z_dim: Dimension of latent vector z
            conditional: If True, condition on letter class
            num_classes: Number of letter classes (26)
        """
        super().__init__()
        self.z_dim = z_dim
        self.conditional = conditional
        
        # Calculate input dimension
        input_dim = z_dim + (num_classes if conditional else 0)
        
        # Project and reshape: z → 7×7×128
        self.project = nn.Sequential(
            nn.Linear(input_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True)
        )
        
        # Upsample: 7×7×128 → 14×14×64 → 28×28×1
        self.main = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, z, class_label=None):
        """
        Generate images from latent code.
        """
        if self.conditional:
            if class_label is None:
                raise ValueError("Class label must be provided for conditional generation.")
            input_vec = torch.cat([z, class_label], 1)
        else:
            input_vec = z
        
        x = self.project(input_vec)
        x = x.view(-1, 128, 7, 7)
        return self.main(x)

class Discriminator(nn.Module):
    def __init__(self, conditional=False, num_classes=26):
        """
        Discriminator network that classifies 28×28 images as real/fake.
        """
        super().__init__()
        self.conditional = conditional
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )
        
        feature_dim = 256 * 3 * 3
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim + (num_classes if conditional else 0), 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, class_label=None):
        """
        Classify images as real (1) or fake (0).
        """
        x = self.features(img)
        x = x.view(x.size(0), -1)
        
        if self.conditional:
            if class_label is None:
                raise ValueError("Class label must be provided for conditional discrimination.")
            x = torch.cat([x, class_label], 1)
            
        return self.classifier(x)