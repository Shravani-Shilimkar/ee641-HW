import torch
import torch.nn as nn

class SSDDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors=3):
        super(SSDDetector, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Backbone
        self.backbone = self._create_backbone()
        
        # Detection Heads
        self.head1 = self._create_head(128, num_anchors, num_classes) # From Block 2
        self.head2 = self._create_head(256, num_anchors, num_classes) # From Block 3
        self.head3 = self._create_head(512, num_anchors, num_classes) # From Block 4
    
    def _create_backbone(self):
        # Block 1 (Stem)
        block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # 224 -> 112
        
        # Block 2
        block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ) # 112 -> 56
        
        # Block 3
        block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) # 56 -> 28
        
        # Block 4
        block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        ) # 28 -> 14
        
        return nn.ModuleList([block1, block2, block3, block4])

    def _create_head(self, in_channels, num_anchors, num_classes):
        # 4 offsets + 1 objectness + num_classes
        out_channels = num_anchors * (5 + num_classes)
        
        head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )
        return head
        
    def forward(self, x):
        features = []
        
        # Pass through backbone
        x = self.backbone[0](x) # Stem
        x = self.backbone[1](x) # Scale 1
        features.append(x)
        x = self.backbone[2](x) # Scale 2
        features.append(x)
        x = self.backbone[3](x) # Scale 3
        features.append(x)

        # Pass through detection heads
        outputs = []
        heads = [self.head1, self.head2, self.head3]
        
        for i, feature in enumerate(features):
            out = heads[i](feature)
            # Reshape: [B, C, H, W] -> [B, H*W*num_anchors, 5 + num_classes]
            B, _, H, W = out.shape
            out = out.permute(0, 2, 3, 1).contiguous()
            out = out.view(B, -1, 5 + self.num_classes)
            outputs.append(out)

        # Concatenate predictions from all scales
        return torch.cat(outputs, dim=1)