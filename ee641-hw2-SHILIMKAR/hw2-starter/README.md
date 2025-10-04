EE 641 - HW2 Starter Code
=========================

Structure:
- `setup_data.py` - Generate datasets (run with --seed 641)
- `provided/` - Utility functions (metrics, visualization)
- `problem1/` - GAN skeleton code
- `problem2/` - VAE skeleton code

See assignment page for full instructions.








## EE 641: Deep Learning - Homework 2

# Personal Information

  Name: Shravani Shilimkar
  USC Email: shilimka@usc.edu


## Problem 1:

Instructions to Run
The experiments are run from the problem1 directory. All required output files will be generated automatically.

Set up the data: From the root directory (ee641-hw2-SHILIMKAR/hw2-starter), run the setup script first:

Bash
python setup_data.py --seed 641
Navigate to the problem directory:

Bash
cd problem1
Run the Vanilla GAN experiment: This is the default setting in train.py. It trains the standard GAN and saves all results to a results folder.

Bash
python train.py
Rename the results folder: To preserve the vanilla results before running the next experiment, rename the folder:

Bash
mv results results_vanilla
Run the "Fixed" GAN experiment:

Open train.py and change the experiment setting from 'vanilla' to 'fixed'.
Run the training script again. This will use the Feature Matching fix and save outputs to a new results folder.

Bash
python train.py
Generate Interpolation Sequences: After each training run, the evaluate.py script can be run to generate an interpolation sequence for the corresponding trained model.

Bash
python evaluate.py

Implementation Notes
GAN Architecture: The Generator uses ConvTranspose2d layers for upsampling, while the Discriminator uses Conv2d layers for feature extraction, as implemented in models.py.

Mode Collapse Fix: The stabilization technique chosen and implemented is Feature Matching. This method modifies the generator's loss function to minimize the L2 distance between the mean feature activations of real and fake samples from an intermediate layer of the discriminator.

Mode Coverage Analysis: The analysis scripts (training_dynamics.py and train.py) rely on a helper function (get_letter_classifier) to classify generated images. For this project, a simple placeholder classifier is used, as the primary goal is to track the number of unique modes generated, not the precise accuracy of their classification.

Experimental Outcome: A key finding from the provided training logs was that the Feature Matching technique, with the default hyperparameters, did not successfully mitigate mode collapse. The final mode coverage for the "fixed" GAN was 3.8%, identical to that of the "vanilla" GAN. The analysis report discusses potential reasons for this outcome, such as hyperparameter sensitivity, which is common in GAN training.




## Problem 2:

#Instructions to Run

Setup: Open a new terminal in VS Code (`Terminal > New Terminal`) and navigate to the project's root `hw2-starter` directory.

bash
cd path/to/your/ee641-hw2-SHILIMKAR/hw2-starter

1.  Train Model:
    Run the training script from the terminal. This will create the `results/` folder with the trained model and logs.

    bash
    python problem2/train.py
    

2.  Run Experiments:
    Open the `problem2/experiments.ipynb` file in the VS Code editor. Run the cells individually by pressing `Shift + Enter` to generate the final analysis and plots.



#Implementation Notes

  Framework: The model is implemented in PyTorch.
  Architecture: A Hierarchical VAE is used to disentangle high-level musical style from low-level rhythmic patterns.
  Training: A cyclical annealing schedule is used to stabilize training and prevent posterior collapse, ensuring the latent variables learn meaningful features.
