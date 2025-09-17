import torch
import cv2
import numpy as np
import os
import json
from torchvision.ops import nms
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from dataset import ShapeDetectionDataset
from model import SSDDetector
from utils import generate_anchors, decode_boxes

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'results/best_model.pth'
VIS_DIR = 'results/visualizations'
NUM_VIS_IMAGES = 10
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
NUM_CLASSES = 3
IMAGE_SIZE = 224
CLASSES = {0: 'circle', 1: 'square', 2: 'triangle'}
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)] # BGR for OpenCV

# Create directories
os.makedirs(VIS_DIR, exist_ok=True)

def post_process(predictions, anchors, conf_thresh, nms_thresh):
    """Applies post-processing steps: decoding, confidence filtering, and NMS."""
    # Split predictions
    pred_loc = predictions[:, :, :4]
    pred_obj = torch.sigmoid(predictions[:, :, 4])
    pred_cls = torch.softmax(predictions[:, :, 5:], dim=-1)

    # Decode boxes
    decoded_boxes = decode_boxes(pred_loc.squeeze(0), anchors)

    # Filter by confidence
    scores, labels = pred_cls.squeeze(0).max(dim=1)
    scores *= pred_obj.squeeze(0)
    
    keep = scores > conf_thresh
    boxes = decoded_boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # Apply NMS
    keep_nms = nms(boxes, scores, nms_thresh)
    
    return boxes[keep_nms], scores[keep_nms], labels[keep_nms]


def visualize_detections(image, pred_boxes, pred_labels, gt_boxes, gt_labels, save_path):
    """Draws predicted and ground-truth boxes on an image."""
    img_np = image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    img_np = img_np.astype(np.uint8).copy()

    # Draw GT boxes (Green)
    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_np, f'GT: {CLASSES[label.item()]}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw Predicted boxes (Blue)
    for box, label in zip(pred_boxes, pred_labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img_np, f'Pred: {CLASSES[label.item()]}', (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    cv2.imwrite(save_path, img_np)

def visualize_anchors(anchors, splits, feature_map_sizes, save_path):
    """Visualizes the anchor boxes for each scale."""
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Anchor Coverage Visualization', fontsize=16)

    start = 0
    for i, (split, f_size) in enumerate(zip(splits, feature_map_sizes)):
        end = start + split
        scale_anchors = anchors[start:end]
        start = end

        ax = axs[i]
        ax.set_title(f'Scale {i+1} ({f_size}x{f_size} Feature Map)')
        ax.set_xlim(0, IMAGE_SIZE)
        ax.set_ylim(IMAGE_SIZE, 0)
        
        # Draw a subset of anchors for clarity
        indices = torch.randint(0, len(scale_anchors), (200,))
        for idx in indices:
            x1, y1, x2, y2 = scale_anchors[idx]
            w, h = x2 - x1, y2 - y1
            ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, edgecolor='r', linewidth=0.5))
    
    plt.savefig(save_path)
    plt.close()

# def analyze_scale_specialization(model, data_loader, anchors, device):
#     """Analyzes which scales detect which object sizes."""
#     model.eval()
#     scale_detections = {0: [], 1: [], 2: []} # Keys: GT class IDs
#     scale_origins = {
#         'scale1': 56*56*3,
#         'scale2': 56*56*3 + 28*28*3
#     }
    
#     with torch.no_grad():
#         for images, targets in tqdm(data_loader, desc="Analyzing Scales"):
#             images = images.to(device)
#             predictions = model(images)
            
#             for i in range(len(targets)):
#                 gt_boxes = targets[i]['boxes']
#                 gt_labels = targets[i]['labels']
#                 pred_boxes, _, _ = post_process(predictions[i:i+1], anchors, CONF_THRESHOLD, NMS_THRESHOLD)
                
#                 if len(pred_boxes) == 0 or len(gt_boxes) == 0:
#                     continue
                
#                 iou_matrix = box_iou(pred_boxes, gt_boxes)
#                 max_iou, max_idx = iou_matrix.max(dim=0)
                
#                 for gt_idx, pred_idx in enumerate(max_idx):
#                     if max_iou[gt_idx] > 0.5:
#                         gt_label = gt_labels[gt_idx].item()
                        
#                         # Find which anchor generated this prediction
#                         original_anchor_idx = (torch.sigmoid(predictions[i, :, 4]) > CONF_THRESHOLD).nonzero(as_tuple=True)[0][pred_idx]
                        
#                         scale = -1
#                         if original_anchor_idx < scale_origins['scale1']:
#                             scale = 1
#                         elif original_anchor_idx < scale_origins['scale2']:
#                             scale = 2
#                         else:
#                             scale = 3
                            
#                         scale_detections[gt_label].append(scale)

#     # Plotting
#     fig, ax = plt.subplots(figsize=(10, 6))
#     class_names = [CLASSES[i] for i in range(NUM_CLASSES)]
#     scale1_counts = [scale_detections[i].count(1) for i in range(NUM_CLASSES)]
#     scale2_counts = [scale_detections[i].count(2) for i in range(NUM_CLASSES)]
#     scale3_counts = [scale_detections[i].count(3) for i in range(NUM_CLASSES)]
    
#     bar_width = 0.25
#     r1 = np.arange(len(class_names))
#     r2 = [x + bar_width for x in r1]
#     r3 = [x + bar_width for x in r2]

#     ax.bar(r1, scale1_counts, color='c', width=bar_width, edgecolor='grey', label='Scale 1 (Small Anchors)')
#     ax.bar(r2, scale2_counts, color='m', width=bar_width, edgecolor='grey', label='Scale 2 (Medium Anchors)')
#     ax.bar(r3, scale3_counts, color='y', width=bar_width, edgecolor='grey', label='Scale 3 (Large Anchors)')
    
#     ax.set_xlabel('Object Class (Size)', fontweight='bold')
#     ax.set_ylabel('Number of Correct Detections', fontweight='bold')
#     ax.set_title('Detection Scale Specialization', fontweight='bold')
#     ax.set_xticks([r + bar_width for r in range(len(class_names))])
#     ax.set_xticklabels(class_names)
#     ax.legend()
#     plt.savefig(os.path.join(VIS_DIR, 'scale_specialization.png'))
#     plt.close()


# Replace the old function with this new one
def analyze_scale_specialization(model, data_loader, anchors, device):
    """Analyzes which scales detect which object sizes."""
    model.eval()
    scale_detections = {0: [], 1: [], 2: []} # Keys: GT class IDs
    
    # Calculate the anchor index boundaries for each scale
    num_anchors_scale1 = 56 * 56 * 3
    num_anchors_scale2 = 28 * 28 * 3
    scale_origins = {
        'scale1_end': num_anchors_scale1,
        'scale2_end': num_anchors_scale1 + num_anchors_scale2
    }
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Analyzing Scales"):
            images = images.to(device)
            predictions = model(images)
            
            # The original code had an unnecessary inner loop here.
            # We now process the batch directly.
            gt_boxes = targets['boxes'][0].to(device) # Get the first (and only) item from the batch
            gt_labels = targets['labels'][0].to(device)
            
            # Post-process the predictions for the single image in the batch
            pred_boxes, _, pred_labels = post_process(predictions, anchors, CONF_THRESHOLD, NMS_THRESHOLD)
            
            if len(pred_boxes) == 0 or len(gt_boxes) == 0:
                continue
            
            # Find the best prediction for each ground truth box
            iou_matrix = box_iou(pred_boxes, gt_boxes)
            if iou_matrix.numel() == 0:
                continue
            max_iou_per_gt, max_idx_per_gt = iou_matrix.max(dim=0)
            
            # For each GT box that was successfully detected, find which scale detected it
            for gt_idx, pred_idx in enumerate(max_idx_per_gt):
                if max_iou_per_gt[gt_idx] > 0.5:
                    gt_label = gt_labels[gt_idx].item()
                    
                    # Find which anchor generated this prediction by looking at the raw model output
                    # This is a bit complex, but it traces the prediction back to its original anchor index
                    conf_scores = torch.sigmoid(predictions[0, :, 4])
                    class_scores, _ = torch.softmax(predictions[0, :, 5:], dim=-1).max(dim=-1)
                    total_scores = conf_scores * class_scores
                    
                    # Get the indices of anchors that passed the confidence threshold
                    candidate_indices = (total_scores > CONF_THRESHOLD).nonzero(as_tuple=True)[0]
                    
                    # The nms operation returns indices relative to the filtered boxes,
                    # so we need to map back to the original anchor indices
                    if pred_idx < len(candidate_indices):
                        original_anchor_idx = candidate_indices[pred_idx]
                        
                        # Determine the scale based on the anchor index
                        scale = -1
                        if original_anchor_idx < scale_origins['scale1_end']:
                            scale = 1
                        elif original_anchor_idx < scale_origins['scale2_end']:
                            scale = 2
                        else:
                            scale = 3
                            
                        if scale != -1:
                            scale_detections[gt_label].append(scale)

    # --- Plotting (This part remains the same) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    class_names = [CLASSES[i] for i in range(NUM_CLASSES)]
    scale1_counts = [scale_detections[i].count(1) for i in range(NUM_CLASSES)]
    scale2_counts = [scale_detections[i].count(2) for i in range(NUM_CLASSES)]
    scale3_counts = [scale_detections[i].count(3) for i in range(NUM_CLASSES)]
    
    bar_width = 0.25
    r1 = np.arange(len(class_names))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    ax.bar(r1, scale1_counts, color='c', width=bar_width, edgecolor='grey', label='Scale 1 (Small Anchors)')
    ax.bar(r2, scale2_counts, color='m', width=bar_width, edgecolor='grey', label='Scale 2 (Medium Anchors)')
    ax.bar(r3, scale3_counts, color='y', width=bar_width, edgecolor='grey', label='Scale 3 (Large Anchors)')
    
    ax.set_xlabel('Object Class (Size)', fontweight='bold')
    ax.set_ylabel('Number of Correct Detections', fontweight='bold')
    ax.set_title('Detection Scale Specialization', fontweight='bold')
    ax.set_xticks([r + bar_width for r in range(len(class_names))])
    ax.set_xticklabels(class_names)
    ax.legend()
    plt.savefig(os.path.join(VIS_DIR, 'scale_specialization.png'))
    plt.close()


def main():
    # --- Load Model and Data ---
    # model = SSDDetector(num_classes=NUM_CLASSES).to(DEVICE)
    # model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    # model.eval()

    # val_dataset = ShapeDetectionDataset(
    #     root_dir='../datasets/problem1/val',
    #     annotation_file='../datasets/problem1/val/annotations.json'
    # )
    # val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = SSDDetector(num_classes=NUM_CLASSES).to(DEVICE)
    # Added weights_only=True to address the warning
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    val_dataset = ShapeDetectionDataset(
        root_dir='../datasets/detection/val',
        annotation_file='../datasets/detection/val_annotations.json'
)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    anchors = generate_anchors(image_size=IMAGE_SIZE).to(DEVICE)

    # --- Generate Visualizations ---
    print(f"Generating detection visualizations for {NUM_VIS_IMAGES} images...")
    for i in range(NUM_VIS_IMAGES):
        image, target = val_dataset[i]
        image_tensor = image.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            predictions = model(image_tensor)
        
        pred_boxes, _, pred_labels = post_process(predictions, anchors, CONF_THRESHOLD, NMS_THRESHOLD)
        
        save_path = os.path.join(VIS_DIR, f'detection_{i}.png')
        visualize_detections(image, pred_boxes.cpu(), pred_labels.cpu(), target['boxes'], target['labels'], save_path)
    print(f"Visualizations saved to {VIS_DIR}")
    
    # --- Visualize Anchor Coverage ---
    print("Generating anchor coverage visualization...")
    splits = [56*56*3, 28*28*3, 14*14*3]
    visualize_anchors(anchors.cpu(), splits, [56, 28, 14], os.path.join(VIS_DIR, 'anchor_coverage.png'))
    print("Anchor visualization saved.")

    # --- Analyze Scale Specialization ---
    print("Analyzing scale specialization...")
    analyze_scale_specialization(model, val_loader, anchors, DEVICE)
    print("Scale analysis saved.")


if __name__ == '__main__':
    main()