# """
# Visualization utilities for gridworld and policies.
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import colors
# from typing import Optional, Tuple
# import os


# class GridWorldVisualizer:
#     """
#     Visualizer for gridworld environment, value functions, and policies.
#     """

#     def __init__(self, grid_size: int = 5):
#         """
#         Initialize visualizer.

#         Args:
#             grid_size: Size of grid
#         """
#         self.grid_size = grid_size

#         # Define special positions
#         self.start_pos = (0, 0)
#         self.goal_pos = (4, 4)
#         self.obstacles = [(1, 2), (2, 1)]
#         self.penalties = [(3, 3), (3, 0)]

#     def plot_value_function(self, values: np.ndarray, title: str = "Value Function") -> None:
#         """
#         Plot value function as heatmap.

#         Args:
#             values: Value function V(s) for each state
#             title: Plot title
#         """
#         # TODO: Reshape values to 2D grid
#         # TODO: Create heatmap with appropriate colormap
#         # TODO: Mark special cells (start, goal, obstacles, penalties)
#         # TODO: Add colorbar and labels
#         # TODO: Save figure to results/visualizations/

#         raise NotImplementedError

#     def plot_policy(self, policy: np.ndarray, title: str = "Optimal Policy") -> None:
#         """
#         Plot policy with arrows showing optimal actions.

#         Args:
#             policy: Array of optimal actions for each state
#             title: Plot title
#         """
#         # TODO: Create grid plot
#         # TODO: For each state:
#         #       - Draw arrow indicating action direction
#         #       - Handle special cells appropriately
#         # TODO: Mark start, goal, obstacles, penalties
#         # TODO: Save figure to results/visualizations/

#         raise NotImplementedError

#     def plot_q_function(self, q_values: np.ndarray, title: str = "Q-Function") -> None:
#         """
#         Plot Q-function with multiple subplots for each action.

#         Args:
#             q_values: Q-function Q(s,a)
#             title: Plot title
#         """
#         # TODO: Create subplot for each action
#         # TODO: For each action:
#         #       - Show Q-values as heatmap
#         #       - Mark special cells
#         # TODO: Add overall title and save

#         raise NotImplementedError

#     def plot_convergence(self, vi_history: list, qi_history: list) -> None:
#         """
#         Plot convergence curves for both algorithms.

#         Args:
#             vi_history: Value iteration convergence history
#             qi_history: Q-iteration convergence history
#         """
#         # TODO: Plot Bellman error vs iteration for both algorithms
#         # TODO: Use log scale for y-axis
#         # TODO: Add legend and labels
#         # TODO: Save figure

#         raise NotImplementedError

#     def create_comparison_figure(self, vi_values: np.ndarray, qi_values: np.ndarray,
#                                 vi_policy: np.ndarray, qi_policy: np.ndarray) -> None:
#         """
#         Create comparison figure showing both algorithms' results.

#         Args:
#             vi_values: Value function from Value Iteration
#             qi_values: Value function from Q-Iteration
#             vi_policy: Policy from Value Iteration
#             qi_policy: Policy from Q-Iteration
#         """
#         # TODO: Create 2x2 subplot
#         #       - Top left: VI value function
#         #       - Top right: QI value function
#         #       - Bottom left: VI policy
#         #       - Bottom right: QI policy
#         # TODO: Highlight any differences
#         # TODO: Save comprehensive comparison figure

#         raise NotImplementedError


# def visualize_results():
#     """
#     Load and visualize saved results from training.
#     """
#     # TODO: Load saved value functions and policies
#     # TODO: Create visualizer instance
#     # TODO: Generate all visualization plots
#     # TODO: Print summary statistics

#     raise NotImplementedError


# if __name__ == '__main__':
#     visualize_results()


"""
Visualization utilities for gridworld and policies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from typing import Optional, Tuple
import os
import json

# Define constants (should match train.py)
RESULTS_DIR = 'results'
VI_RESULTS_FILE = os.path.join(RESULTS_DIR, 'value_iteration_results.json')
QI_RESULTS_FILE = os.path.join(RESULTS_DIR, 'q_iteration_results.json')
VI_POLICY_FILE = os.path.join(RESULTS_DIR, 'vi_optimal_policy.npy')
QI_POLICY_FILE = os.path.join(RESULTS_DIR, 'qi_optimal_policy.npy')
VISUALIZATION_DIR = os.path.join(RESULTS_DIR, 'visualizations')

class GridWorldVisualizer:
    """
    Visualizer for gridworld environment, value functions, and policies.
    """

    def __init__(self, grid_size: int = 5):
        """
        Initialize visualizer.

        Args:
            grid_size: Size of grid
        """
        self.grid_size = grid_size

        # Define special positions (row, col)
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        self.obstacles = [(1, 2), (2, 1)]
        self.penalties = [(3, 3), (3, 0)]
        
        # Action map for plotting arrows: (dx, dy) where dx is movement along x (col), dy along y (row)
        # 0=UP (-1, 0), 1=RIGHT (0, 1), 2=DOWN (1, 0), 3=LEFT (0, -1)
        self.action_vectors = {
            0: (0, 0.4),    # UP
            1: (0.4, 0),    # RIGHT
            2: (0, -0.4),   # DOWN
            3: (-0.4, 0)    # LEFT
        }
        self.action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)


    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Helper to convert position to state index."""
        row, col = pos
        return row * self.grid_size + col
        
    def _draw_grid_overlay(self, ax, V_grid: np.ndarray):
        """Helper to draw grid lines, labels, and special cells."""
        # Draw grid lines
        ax.set_xticks(np.arange(self.grid_size + 1) - 0.5, minor=False)
        ax.set_yticks(np.arange(self.grid_size + 1) - 0.5, minor=False)
        ax.grid(which='major', color='black', linestyle='-', linewidth=1)
        
        # Set tick labels (col and row indices)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
        # Invert y-axis to match (row, col) convention where (0,0) is top-left
        ax.invert_yaxis()
        
        # Mark special cells
        # Note: In matplotlib image coordinates, Y is row, X is column
        for r, c in self.obstacles:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='lightgray', alpha=0.5))
            ax.text(c, r, 'X', color='black', ha='center', va='center', fontsize=12, fontweight='bold')
        
        for r, c in self.penalties:
            ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='salmon', alpha=0.3))
            ax.text(c, r, 'P (-5)', color='black', ha='center', va='center', fontsize=8)
            
        r, c = self.goal_pos
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='green', alpha=0.3))
        ax.text(c, r, 'G (+10)', color='darkgreen', ha='center', va='center', fontsize=10, fontweight='bold')
        
        r, c = self.start_pos
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='skyblue', alpha=0.3))
        ax.text(c, r, 'S', color='darkblue', ha='center', va='center', fontsize=12, fontweight='bold')

    def plot_value_function(self, values: np.ndarray, title: str = "Value Function", filename: str = "value_function.png") -> None:
        """
        Plot value function as heatmap.

        Args:
            values: Value function V(s) for each state
            title: Plot title
            filename: Name to save the figure
        """
        V_grid = values.reshape(self.grid_size, self.grid_size)
        
        plt.figure(figsize=(6, 6))
        ax = plt.gca()

        # Create heatmap
        im = ax.imshow(V_grid, cmap='viridis')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Optimal Value $V^*(s)$')

        self._draw_grid_overlay(ax, V_grid)
        
        # Add value numbers to each cell
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                ax.text(c, r, f'{V_grid[r, c]:.2f}', 
                        ha='center', va='center', color='w', fontsize=8, fontweight='bold')

        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, filename))
        plt.close()

    def plot_policy(self, policy: np.ndarray, title: str = "Optimal Policy", filename: str = "policy.png") -> None:
        """
        Plot policy with arrows showing optimal actions.

        Args:
            policy: Array of optimal actions for each state
            title: Plot title
            filename: Name to save the figure
        """
        pi_grid = policy.reshape(self.grid_size, self.grid_size)

        plt.figure(figsize=(6, 6))
        ax = plt.gca()
        
        # Create a blank background for the grid
        ax.imshow(np.zeros((self.grid_size, self.grid_size)), cmap='gray_r', alpha=0.1)

        self._draw_grid_overlay(ax, pi_grid) # Use pi_grid just for size/labels

        # Plot arrows
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                state = self._pos_to_state((r, c))
                action = pi_grid[r, c]
                
                # Check if it's a terminal state (marked as -1 or goal pos)
                if (r, c) == self.goal_pos or action == -1:
                    continue
                
                # Center of the cell (x=col, y=row)
                center_x, center_y = c, r
                
                # Get action vector
                dx, dy = self.action_vectors[action]
                
                # Draw arrow (quiver: X, Y, U, V)
                # U is the change in X (col), V is the change in Y (row)
                # Note: Matplotlib's y-axis is inverted by default, which works with our inverted_yaxis setting.
                ax.quiver(center_x, center_y, dx, -dy, 
                          angles='xy', scale_units='xy', scale=1, 
                          color='darkred', width=0.015, headwidth=5)

        ax.set_title(title)
        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(-0.5, self.grid_size - 0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALIZATION_DIR, filename))
        plt.close()

    def create_comparison_figure(self, vi_values: np.ndarray, qi_values: np.ndarray,
                                vi_policy: np.ndarray, qi_policy: np.ndarray) -> None:
        """
        Create comparison figure showing both algorithms' results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Get overall min/max for consistent colormapping
        all_values = np.concatenate([vi_values, qi_values])
        vmin = all_values.min()
        vmax = all_values.max()

        # --- Top Left: VI Value Function ---
        ax00 = axes[0, 0]
        V_grid_vi = vi_values.reshape(self.grid_size, self.grid_size)
        im00 = ax00.imshow(V_grid_vi, cmap='viridis', vmin=vmin, vmax=vmax)
        self._draw_grid_overlay(ax00, V_grid_vi)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                ax00.text(c, r, f'{V_grid_vi[r, c]:.2f}', ha='center', va='center', color='w', fontsize=8, fontweight='bold')
        ax00.set_title("A) Value Iteration: Optimal Value $V^*(s)$")

        # --- Top Right: QI Value Function ---
        ax01 = axes[0, 1]
        V_grid_qi = qi_values.reshape(self.grid_size, self.grid_size)
        im01 = ax01.imshow(V_grid_qi, cmap='viridis', vmin=vmin, vmax=vmax)
        self._draw_grid_overlay(ax01, V_grid_qi)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                ax01.text(c, r, f'{V_grid_qi[r, c]:.2f}', ha='center', va='center', color='w', fontsize=8, fontweight='bold')
        ax01.set_title("B) Q-Iteration: Optimal Value $V^*(s)$")
        
        # Add a single colorbar for value functions
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.55, 0.03, 0.4]) # [left, bottom, width, height]
        cbar = fig.colorbar(im00, cax=cbar_ax)
        cbar.set_label('Optimal Value $V^*(s)$')
        
        
        # --- Bottom Left: VI Policy ---
        ax10 = axes[1, 0]
        ax10.imshow(np.zeros((self.grid_size, self.grid_size)), cmap='gray_r', alpha=0.1)
        self._draw_grid_overlay(ax10, V_grid_vi)
        pi_grid_vi = vi_policy.reshape(self.grid_size, self.grid_size)
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                action = pi_grid_vi[r, c]
                if (r, c) != self.goal_pos and action != -1:
                    dx, dy = self.action_vectors[action]
                    ax10.quiver(c, r, dx, -dy, angles='xy', scale_units='xy', scale=1, color='darkred', width=0.015, headwidth=5)
        ax10.set_title("C) Value Iteration: Optimal Policy $\pi^*(s)$")
        ax10.set_xlim(-0.5, self.grid_size - 0.5)
        ax10.set_ylim(-0.5, self.grid_size - 0.5)


        # --- Bottom Right: QI Policy ---
        ax11 = axes[1, 1]
        ax11.imshow(np.zeros((self.grid_size, self.grid_size)), cmap='gray_r', alpha=0.1)
        self._draw_grid_overlay(ax11, V_grid_qi)
        pi_grid_qi = qi_policy.reshape(self.grid_size, self.grid_size)
        # Highlight policy differences
        policy_diff = (pi_grid_vi != pi_grid_qi)
        policy_diff_states = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size) if policy_diff[r, c] and (r, c) != self.goal_pos]
        
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                action = pi_grid_qi[r, c]
                if (r, c) != self.goal_pos and action != -1:
                    dx, dy = self.action_vectors[action]
                    color = 'darkred'
                    if (r, c) in policy_diff_states:
                        color = 'blue' # Highlight different policies
                        ax11.add_patch(plt.Circle((c, r), 0.3, color='blue', alpha=0.2, zorder=2))
                    ax11.quiver(c, r, dx, -dy, angles='xy', scale_units='xy', scale=1, color=color, width=0.015, headwidth=5)
        
        ax11.set_title("D) Q-Iteration: Optimal Policy $\pi^*(s)$ (Blue = Diff)")
        ax11.set_xlim(-0.5, self.grid_size - 0.5)
        ax11.set_ylim(-0.5, self.grid_size - 0.5)

        plt.suptitle("Comparison of Value Iteration and Q-Iteration Results")
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for colorbar
        plt.savefig(os.path.join(VISUALIZATION_DIR, "vi_qi_comparison.png"))
        plt.close(fig)
        
        if policy_diff_states:
            print(f"Note: Policies differ in {len(policy_diff_states)} state(s). Check Q-values for tie-breaking.")


def visualize_results():
    """
    Load and visualize saved results from training.
    """
    if not os.path.exists(VI_RESULTS_FILE) or not os.path.exists(QI_RESULTS_FILE):
        print("Error: Training results not found. Run train.py first.")
        return

    # 1. Load data
    with open(VI_RESULTS_FILE, 'r') as f:
        vi_results = json.load(f)
    with open(QI_RESULTS_FILE, 'r') as f:
        qi_results = json.load(f)

    vi_values = np.array(vi_results['optimal_value_function'])
    qi_values = np.array(qi_results['optimal_value_function'])
    vi_policy = np.load(VI_POLICY_FILE)
    qi_policy = np.load(QI_POLICY_FILE)

    # 2. Create visualizer instance
    visualizer = GridWorldVisualizer(grid_size=5)

    # 3. Generate visualization plots
    print("Generating visualizations...")
    visualizer.plot_value_function(vi_values, title="Value Iteration: Optimal Value Function $V^*(s)$", filename="vi_value_function.png")
    visualizer.plot_value_function(qi_values, title="Q-Iteration: Optimal Value Function $V^*(s)$", filename="qi_value_function.png")
    visualizer.plot_policy(vi_policy, title="Value Iteration: Optimal Policy $\pi^*(s)$", filename="vi_optimal_policy.png")
    visualizer.plot_policy(qi_policy, title="Q-Iteration: Optimal Policy $\pi^*(s)$", filename="qi_optimal_policy.png")
    
    # 4. Create comprehensive comparison figure
    visualizer.create_comparison_figure(vi_values, qi_values, vi_policy, qi_policy)
    
    # 5. Print summary statistics
    print("\n--- Visualization Summary ---")
    print(f"VI Iterations: {vi_results['n_iterations']}")
    print(f"QI Iterations: {qi_results['n_iterations']}")
    
    # Check if V* is close (tolerance for float comparison)
    v_diff = np.max(np.abs(vi_values - qi_values))
    print(f"Max difference in V*: {v_diff:.6e}")
    
    # Check if policies match (ignoring terminal state)
    policy_match = np.all(vi_policy == qi_policy)
    print(f"Policies Match: {policy_match}")
    
    print(f"\nAll visualizations saved to {VISUALIZATION_DIR}/")


if __name__ == '__main__':
    visualize_results()