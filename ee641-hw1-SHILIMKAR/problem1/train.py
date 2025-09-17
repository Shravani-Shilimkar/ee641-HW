import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm

from dataset import ShapeDetectionDataset
from model import SSDDetector
from utils import generate_anchors, create_ssd_targets, collate_fn
from loss import MultiTaskLoss

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
NUM_CLASSES = 3
IMAGE_SIZE = 224
RESULTS_DIR = 'results'
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, 'best_model.pth')
LOG_FILE_PATH = os.path.join(RESULTS_DIR, 'training_log.json')

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_one_epoch(model, data_loader, optimizer, criterion, anchors, device):
    model.train()
    total_loss = 0
    pbar = tqdm(data_loader, desc="Training")
    for images, targets in pbar:
        images = images.to(device)
        # Generate targets on the fly
        gt_offsets, gt_objectness, gt_classes = create_ssd_targets(anchors, targets)
        gt_offsets = gt_offsets.to(device)
        gt_objectness = gt_objectness.to(device)
        gt_classes = gt_classes.to(device)

        # Forward pass
        predictions = model(images)
        loss, _ = criterion(predictions, gt_offsets, gt_objectness, gt_classes)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return total_loss / len(data_loader)

def validate(model, data_loader, criterion, anchors, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Validating")
        for images, targets in pbar:
            images = images.to(device)
            gt_offsets, gt_objectness, gt_classes = create_ssd_targets(anchors, targets)
            gt_offsets = gt_offsets.to(device)
            gt_objectness = gt_objectness.to(device)
            gt_classes = gt_classes.to(device)

            predictions = model(images)
            loss, _ = criterion(predictions, gt_offsets, gt_objectness, gt_classes)
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

    return total_loss / len(data_loader)

def main():
    print(f"Using device: {DEVICE}")

    # --- Data Loading ---
    # train_dataset = ShapeDetectionDataset(
    #     root_dir='../datasets/problem1/train',
    #     annotation_file='../datasets/problem1/train/annotations.json'
    # )
    # val_dataset = ShapeDetectionDataset(
    #     root_dir='../datasets/problem1/val',
    #     annotation_file='../datasets/problem1/val/annotations.json'
    # )


    # --- Data Loading ---
    train_dataset = ShapeDetectionDataset(
        root_dir='../datasets/detection/train',
        annotation_file='../datasets/detection/train_annotations.json'
    )
    val_dataset = ShapeDetectionDataset(
        root_dir='../datasets/detection/val',
        annotation_file='../datasets/detection/val_annotations.json'
    )


    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    # --- Model, Anchors, Loss, Optimizer ---
    anchors = generate_anchors(image_size=IMAGE_SIZE).to(DEVICE)
    model = SSDDetector(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = MultiTaskLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # --- Training Loop ---
    best_val_loss = float('inf')
    training_log = []

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, anchors, DEVICE)
        val_loss = validate(model, val_loader, criterion, anchors, DEVICE)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        log_entry = {'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss}
        training_log.append(log_entry)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"Saved new best model to {BEST_MODEL_PATH}")

    # Save training log
    with open(LOG_FILE_PATH, 'w') as f:
        json.dump(training_log, f, indent=4)
    print(f"Training log saved to {LOG_FILE_PATH}")

if __name__ == '__main__':
    main()