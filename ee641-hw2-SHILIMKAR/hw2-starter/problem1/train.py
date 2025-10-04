# """
# Main training script for GAN experiments.
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import json
# import os
# from pathlib import Path

# from dataset import FontDataset
# from models import Generator, Discriminator
# from training_dynamics import train_gan, analyze_mode_coverage
# from fixes import train_gan_with_fix

# def main():
#     """
#     Main training entry point for GAN experiments.
#     """
#     # Configuration
#     config = {
#         'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
#         'batch_size': 64,
#         'num_epochs': 100,
#         'z_dim': 100,
#         'learning_rate': 0.0002,
#         'data_dir': 'data/fonts',
#         'checkpoint_dir': 'checkpoints',
#         'results_dir': 'results',
#         'experiment': 'vanilla',  # 'vanilla' or 'fixed'
#         'fix_type': 'feature_matching'  # Used if experiment='fixed'
#     }
    
#     # Create directories
#     Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
#     Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
#     # Initialize dataset and dataloader
#     train_dataset = FontDataset(config['data_dir'], split='train')
#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=config['batch_size'],
#         shuffle=True,
#         num_workers=2
#     )
    
#     # Initialize models
#     generator = Generator(z_dim=config['z_dim']).to(config['device'])
#     discriminator = Discriminator().to(config['device'])
    
#     # Train model
#     if config['experiment'] == 'vanilla':
#         print("Training vanilla GAN (expect mode collapse)...")
#         history = train_gan(
#             generator, 
#             discriminator, 
#             train_loader,
#             num_epochs=config['num_epochs'],
#             device=config['device']
#         )
#     else:
#         print(f"Training GAN with {config['fix_type']} fix...")
#         history = train_gan_with_fix(
#             generator,
#             discriminator, 
#             train_loader,
#             num_epochs=config['num_epochs'],
#             fix_type=config['fix_type']
#         )
    
#     # Save results
#     # TODO: Save training history to JSON
#     with open(f"{config['results_dir']}/training_log.json", 'w') as f:
#         json.dump(history, f, indent=2)
    
#     # TODO: Save final model checkpoint
#     torch.save({
#         'generator_state_dict': generator.state_dict(),
#         'discriminator_state_dict': discriminator.state_dict(),
#         'config': config,
#         'final_epoch': config['num_epochs']
#     }, f"{config['results_dir']}/best_generator.pth")
    
#     print(f"Training complete. Results saved to {config['results_dir']}/")

# if __name__ == '__main__':
#     main()



"""
Main training script for GAN experiments.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

from dataset import FontDataset
from models import Generator, Discriminator
from training_dynamics import train_gan, visualize_mode_collapse, analyze_mode_coverage
from fixes import train_gan_with_fix

def save_mode_histogram(generator, device, save_path):
    """Generates samples and saves a histogram of the letters found."""
    print("Generating final mode coverage histogram...")
    # This can take a moment
    _, letter_counts = analyze_mode_coverage(generator, device, n_samples=2000)
    
    surviving_letters = {k: v for k, v in letter_counts.items() if v > 0}
    if not surviving_letters:
        print("No modes survived, skipping histogram.")
        return

    letters = list(surviving_letters.keys())
    counts = list(surviving_letters.values())
    
    plt.figure(figsize=(12, 6))
    plt.bar(letters, counts)
    plt.title('Final Mode Coverage: Distribution of Generated Letters')
    plt.xlabel('Letter')
    plt.ylabel('Count')
    plt.savefig(save_path)
    plt.close()
    print(f"Histogram saved to {save_path}")

def main():
    """
    Main training entry point for GAN experiments.
    """
    config = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 64,
        'num_epochs': 100,
        'z_dim': 100,
        'learning_rate': 0.0002,
        'data_dir': '../data/fonts',
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results',
        'experiment': 'fixed',  # 'vanilla' or 'fixed'
        'fix_type': 'feature_matching'
    }
    
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
    train_dataset = FontDataset(config['data_dir'], split='train')
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=2
    )
    
    generator = Generator(z_dim=config['z_dim']).to(config['device'])
    discriminator = Discriminator().to(config['device'])
    
    if config['experiment'] == 'vanilla':
        print("Training vanilla GAN (expect mode collapse)...")
        history = train_gan(
            generator, 
            discriminator, 
            train_loader,
            num_epochs=config['num_epochs'],
            device=config['device']
        )
    else:
        print(f"Training GAN with {config['fix_type']} fix...")
        history = train_gan_with_fix(
            generator,
            discriminator, 
            train_loader,
            num_epochs=config['num_epochs'],
            fix_type=config['fix_type'],
            device=config['device']
        )
    
    # Save training history to JSON
    results_file = os.path.join(config['results_dir'], 'training_log.json')
    with open(results_file, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Save final model checkpoint
    model_file = os.path.join(config['results_dir'], 'best_generator.pth')
    torch.save(generator.state_dict(), model_file)

    # Save mode collapse plot
    plot_path = os.path.join(config['results_dir'], 'mode_collapse_analysis.png')
    if history['mode_coverage']:
        visualize_mode_collapse(history, plot_path)
    
    # Save final mode coverage histogram
    histogram_path = os.path.join(config['results_dir'], 'visualizations', 'mode_coverage_histogram.png')
    save_mode_histogram(generator, config['device'], histogram_path)
    
    print(f"\nTraining complete. All results saved to {config['results_dir']}/")

if __name__ == '__main__':
    main()