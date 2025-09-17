import os
import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path
from tqdm import tqdm

def create_stick_figure_data(num_samples, data_dir):
    """Generates stick figure images and annotation JSON file."""
    image_dir = Path(data_dir) / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotations = []

    print(f"Generating {num_samples} samples in {data_dir}...")
    for i in tqdm(range(num_samples)):
        img_size = (128, 128)
        img = Image.new('L', img_size, color=0) # 'L' for grayscale
        draw = ImageDraw.Draw(img)

        # Randomly define the stick figure's core points with some noise
        center_x = np.random.randint(40, 88)
        center_y = np.random.randint(40, 88)
        
        head = (center_x + np.random.randint(-3, 3), center_y - 20 + np.random.randint(-3, 3))
        neck = (center_x, center_y - 15)
        torso_end = (center_x, center_y + 15)
        
        l_hand = (center_x - 20 + np.random.randint(-5, 5), center_y + np.random.randint(-5, 5))
        r_hand = (center_x + 20 + np.random.randint(-5, 5), center_y + np.random.randint(-5, 5))
        l_foot = (center_x - 10 + np.random.randint(-5, 5), center_y + 30 + np.random.randint(-5, 5))
        r_foot = (center_x + 10 + np.random.randint(-5, 5), center_y + 30 + np.random.randint(-5, 5))

        # Define the 5 keypoints to be annotated
        keypoints = [head, l_hand, r_hand, l_foot, r_foot]
        
        # Draw the figure
        draw.line([head, torso_end], fill=255, width=2) # Torso + head
        draw.line([neck, l_hand], fill=255, width=2)   # Left arm
        draw.line([neck, r_hand], fill=255, width=2)   # Right arm
        draw.line([torso_end, l_foot], fill=255, width=2) # Left leg
        draw.line([torso_end, r_foot], fill=255, width=2) # Right leg
        
        # Save image
        image_filename = f"{i:04d}.png"
        img.save(image_dir / image_filename)

        # Add to annotations
        annotations.append({
            "image_path": image_filename,
            "keypoints": keypoints
        })

    # Save annotations file
    with open(Path(data_dir) / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=4)

if __name__ == "__main__":
    # Create training and testing data
    create_stick_figure_data(1000, "data/train")
    create_stick_figure_data(200, "data/test")
    print("\nDataset generation complete! ✅")