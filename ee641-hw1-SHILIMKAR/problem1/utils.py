import torch
import numpy as np
from torchvision.ops import box_iou

def generate_anchors(image_size=224, feature_map_sizes=[56, 28, 14], scales=[[16, 24, 32], [48, 64, 96], [96, 128, 192]], aspect_ratios=[[1.0], [1.0], [1.0]]):
    """
    Generates anchor boxes for each feature map scale.
    """
    all_anchors = []
    for i, fmap_size in enumerate(feature_map_sizes):
        anchors = []
        stride = image_size / fmap_size
        for y in range(fmap_size):
            for x in range(fmap_size):
                cx = (x + 0.5) * stride
                cy = (y + 0.5) * stride
                for scale in scales[i]:
                    for ar in aspect_ratios[i]:
                        h = scale / np.sqrt(ar)
                        w = scale * np.sqrt(ar)
                        
                        x1 = cx - w / 2
                        y1 = cy - h / 2
                        x2 = cx + w / 2
                        y2 = cy + h / 2
                        anchors.append([x1, y1, x2, y2])
        all_anchors.append(torch.tensor(anchors, dtype=torch.float32))
    
    return torch.cat(all_anchors, dim=0)


def match_anchors_to_gt(anchors, gt_boxes, iou_threshold_pos=0.7, iou_threshold_neg=0.3):
    """
    Matches anchors to ground-truth boxes based on IoU.
    Returns:
        - matched_gt_idx: Index of the matched GT box for each anchor.
        - labels: -1 for ignore, 0 for background, 1 for foreground.
    """
    if len(gt_boxes) == 0:
        return torch.full((len(anchors),), -1, dtype=torch.long), \
               torch.zeros(len(anchors), dtype=torch.long)

    iou_matrix = box_iou(anchors, gt_boxes) # [num_anchors, num_gt_boxes]
    
    # For each GT box, find the anchor with the highest IoU
    max_iou_per_gt, max_iou_per_gt_idx = iou_matrix.max(dim=0)
    
    # For each anchor, find the GT box with the highest IoU
    max_iou_per_anchor, max_iou_per_anchor_idx = iou_matrix.max(dim=1)
    
    labels = torch.full((len(anchors),), -1, dtype=torch.long) # -1: ignore
    labels[max_iou_per_anchor < iou_threshold_neg] = 0 # 0: background
    
    # Assign positive label to anchors with highest IoU for each GT box
    labels[max_iou_per_gt_idx] = 1 # 1: foreground
    
    # Assign positive label to anchors with IoU > threshold
    labels[max_iou_per_anchor >= iou_threshold_pos] = 1
    
    matched_gt_idx = max_iou_per_anchor_idx
    
    return matched_gt_idx, labels


def encode_boxes(anchors, gt_boxes):
    """Encodes ground-truth boxes to the anchor-relative offset format (tx, ty, tw, th)."""
    # Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h]
    anc_w = anchors[:, 2] - anchors[:, 0]
    anc_h = anchors[:, 3] - anchors[:, 1]
    anc_cx = anchors[:, 0] + 0.5 * anc_w
    anc_cy = anchors[:, 1] + 0.5 * anc_h

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_cx = gt_boxes[:, 0] + 0.5 * gt_w
    gt_cy = gt_boxes[:, 1] + 0.5 * gt_h

    # Encode
    tx = (gt_cx - anc_cx) / anc_w
    ty = (gt_cy - anc_cy) / anc_h
    tw = torch.log(gt_w / anc_w)
    th = torch.log(gt_h / anc_h)

    return torch.stack([tx, ty, tw, th], dim=1)


def decode_boxes(preds, anchors):
    """Decodes predicted offsets back to bounding box coordinates."""
    anc_w = anchors[:, 2] - anchors[:, 0]
    anc_h = anchors[:, 3] - anchors[:, 1]
    anc_cx = anchors[:, 0] + 0.5 * anc_w
    anc_cy = anchors[:, 1] + 0.5 * anc_h

    pred_cx = preds[:, 0] * anc_w + anc_cx
    pred_cy = preds[:, 1] * anc_h + anc_cy
    pred_w = torch.exp(preds[:, 2]) * anc_w
    pred_h = torch.exp(preds[:, 3]) * anc_h

    pred_x1 = pred_cx - 0.5 * pred_w
    pred_y1 = pred_cy - 0.5 * pred_h
    pred_x2 = pred_cx + 0.5 * pred_w
    pred_y2 = pred_cy + 0.5 * pred_h

    return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)


def create_ssd_targets(anchors, targets):
    """
    Creates the ground truth targets for the SSD model for a batch.
    """
    batch_size = len(targets)
    num_anchors = anchors.shape[0]
    num_classes = 3 
    
    # Initialize target tensors
    batch_gt_offsets = torch.zeros(batch_size, num_anchors, 4)
    batch_gt_objectness = torch.zeros(batch_size, num_anchors)
    batch_gt_classes = torch.zeros(batch_size, num_anchors, dtype=torch.long)

    for i in range(batch_size):
        gt_boxes = targets[i]['boxes']
        gt_labels = targets[i]['labels']
        
        if len(gt_boxes) > 0:
            matched_gt_idx, anchor_labels = match_anchors_to_gt(anchors, gt_boxes)
            
            # Positive anchors
            pos_mask = (anchor_labels == 1)
            if pos_mask.any():
                matched_boxes = gt_boxes[matched_gt_idx[pos_mask]]
                encoded_boxes = encode_boxes(anchors[pos_mask], matched_boxes)
                batch_gt_offsets[i, pos_mask] = encoded_boxes
                batch_gt_classes[i, pos_mask] = gt_labels[matched_gt_idx[pos_mask]]
            
            # Objectness labels (1 for pos, 0 for neg)
            batch_gt_objectness[i, pos_mask] = 1.0
            # anchor_labels is already 0 for negative anchors
            
    return batch_gt_offsets, batch_gt_objectness, batch_gt_classes


def collate_fn(batch):
    """Custom collate function for the DataLoader."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    images = torch.stack(images, 0)
    return images, targets