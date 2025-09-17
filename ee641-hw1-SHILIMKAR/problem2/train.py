# problem2/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json
import os
from tqdm import tqdm

from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet

def train_model(model, model_name, train_loader, val_loader, num_epochs, device):
    """Generic training function for a model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # Training loop
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
        
        epoch_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        # Save the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), f"results/{model_name}.pth")
            print(f"Saved best model {model_name}.pth with val loss: {best_val_loss:.6f}")
            
    return history

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs("results", exist_ok=True)
    
    # --- Train Heatmap Model ---
    print("\n--- Training HeatmapNet ---")
    heatmap_dataset = KeypointDataset(
        image_dir='data/train/images', 
        annotation_file='data/train/annotations.json', 
        output_type='heatmap'
    )
    train_size = int(0.8 * len(heatmap_dataset))
    val_size = len(heatmap_dataset) - train_size
    train_hm_dataset, val_hm_dataset = random_split(heatmap_dataset, [train_size, val_size])
    
    train_hm_loader = DataLoader(train_hm_dataset, batch_size=32, shuffle=True)
    val_hm_loader = DataLoader(val_hm_dataset, batch_size=32, shuffle=False)
    
    heatmap_model = HeatmapNet().to(device)
    hm_history = train_model(heatmap_model, 'heatmap_model', train_hm_loader, val_hm_loader, 30, device)

    # --- Train Regression Model ---
    print("\n--- Training RegressionNet ---")
    regression_dataset = KeypointDataset(
        image_dir='data/train/images', 
        annotation_file='data/train/annotations.json', 
        output_type='regression'
    )
    train_reg_dataset, val_reg_dataset = random_split(regression_dataset, [train_size, val_size])
    
    train_reg_loader = DataLoader(train_reg_dataset, batch_size=32, shuffle=True)
    val_reg_loader = DataLoader(val_reg_dataset, batch_size=32, shuffle=False)

    regression_model = RegressionNet().to(device)
    reg_history = train_model(regression_model, 'regression_model', train_reg_loader, val_reg_loader, 30, device)

    # --- Save Training Logs ---
    training_log = {
        'heatmap': hm_history,
        'regression': reg_history
    }
    with open('results/training_log.json', 'w') as f:
        json.dump(training_log, f, indent=4)
    print("\nTraining complete. Models and logs saved in 'results/'.")

if __name__ == '__main__':
    main()