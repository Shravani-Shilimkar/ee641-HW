import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, loc_weight=2.0, cls_weight=1.0, obj_weight=1.0):
        super(MultiTaskLoss, self).__init__()
        self.loc_weight = loc_weight
        self.cls_weight = cls_weight
        self.obj_weight = obj_weight
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, predictions, targets_loc, targets_obj, targets_cls):
        """
        Args:
            predictions (Tensor): [B, num_anchors, 5 + num_classes]
            targets_loc (Tensor): [B, num_anchors, 4]
            targets_obj (Tensor): [B, num_anchors]
            targets_cls (Tensor): [B, num_anchors]
        """
        # Split predictions
        pred_loc = predictions[:, :, :4]
        pred_obj = predictions[:, :, 4]
        pred_cls = predictions[:, :, 5:]

        # Identify positive and negative anchors
        pos_mask = (targets_obj > 0)
        neg_mask = (targets_obj == 0)
        
        num_pos = pos_mask.sum().clamp(min=1).float()

        # 1. Localization Loss (Smooth L1) - only for positive anchors
        loss_loc = self.smooth_l1_loss(pred_loc[pos_mask], targets_loc[pos_mask])
        
        # 2. Classification Loss (Cross Entropy) - only for positive anchors
        # Reshape for CrossEntropyLoss: [num_pos, num_classes]
        loss_cls = self.cross_entropy_loss(pred_cls[pos_mask], targets_cls[pos_mask])

        # 3. Objectness Loss (BCE) - for all positive and negative anchors
        loss_obj = self.bce_with_logits_loss(pred_obj[pos_mask | neg_mask], targets_obj[pos_mask | neg_mask])

        # Normalize and combine losses
        total_loss = (self.loc_weight * loss_loc + self.cls_weight * loss_cls + self.obj_weight * loss_obj) / num_pos
        
        return total_loss, (loss_loc / num_pos, loss_cls / num_pos, loss_obj / num_pos)