# """
# Evaluation script for trained multi-agent models.
# """

# import torch
# import torch.nn as nn
# import numpy as np
# import matplotlib.pyplot as plt
# from typing import Tuple, List, Dict
# import json
# import os
# from multi_agent_env import MultiAgentEnv
# from models import AgentDQN


# class MultiAgentEvaluator:
#     """
#     Evaluator for analyzing trained multi-agent policies.
#     """

#     def __init__(self, env: MultiAgentEnv, model_A: nn.Module, model_B: nn.Module):
#         """
#         Initialize evaluator.

#         Args:
#             env: Multi-agent environment
#             model_A: Trained model for Agent A
#             model_B: Trained model for Agent B
#         """
#         self.env = env
#         self.model_A = model_A
#         self.model_B = model_B
#         # Use CPU for small networks
#         self.device = torch.device("cpu")

#         # Move models to device and set to evaluation mode
#         self.model_A.to(self.device)
#         self.model_B.to(self.device)
#         self.model_A.eval()
#         self.model_B.eval()

#     def run_episode(self, render: bool = False) -> Tuple[float, bool, Dict]:
#         """
#         Run single evaluation episode.

#         Args:
#             render: Whether to render environment

#         Returns:
#             reward: Episode reward
#             success: Whether target was reached
#             info: Episode statistics
#         """
#         # TODO: Reset environment
#         # TODO: Initialize episode tracking
#         # TODO: Run episode with greedy policy
#         # TODO: Track communication patterns
#         # TODO: Return results and statistics

#         raise NotImplementedError

#     def evaluate_performance(self, num_episodes: int = 100) -> Dict:
#         """
#         Evaluate overall performance statistics.

#         Args:
#             num_episodes: Number of evaluation episodes

#         Returns:
#             Statistics dictionary
#         """
#         # TODO: Run multiple episodes
#         # TODO: Compute success rate
#         # TODO: Analyze path lengths
#         # TODO: Measure coordination efficiency
#         # TODO: Return comprehensive statistics

#         raise NotImplementedError

#     def analyze_communication(self, num_episodes: int = 20) -> Dict:
#         """
#         Analyze emergent communication protocols.

#         Returns:
#             Communication analysis results
#         """
#         # TODO: Track communication signals over episodes
#         # TODO: Analyze signal patterns (magnitude, variance, correlation)
#         # TODO: Identify communication strategies
#         # TODO: Return analysis results

#         raise NotImplementedError

#     def visualize_trajectory(self, save_path: str = 'results/trajectory.png') -> None:
#         """
#         Visualize agent trajectories in an episode.

#         Args:
#             save_path: Path to save visualization
#         """
#         # TODO: Run episode while tracking positions
#         # TODO: Create grid visualization
#         # TODO: Plot agent paths
#         # TODO: Mark key events (near target, coordination points)
#         # TODO: Save figure

#         raise NotImplementedError

#     def plot_communication_heatmap(self, save_path: str = 'results/comm_heatmap.png') -> None:
#         """
#         Create heatmap of communication signals across grid positions.

#         Args:
#             save_path: Path to save figure
#         """
#         # TODO: Sample communication signals at each grid position
#         # TODO: Create heatmaps for both agents
#         # TODO: Show correlation with distance to target
#         # TODO: Save visualization

#         raise NotImplementedError

#     def test_generalization(self, num_configs: int = 10) -> Dict:
#         """
#         Test generalization to new environment configurations.

#         Args:
#             num_configs: Number of test configurations

#         Returns:
#             Generalization performance statistics
#         """
#         # TODO: Generate new obstacle configurations
#         # TODO: Test performance on each configuration
#         # TODO: Compare to training performance
#         # TODO: Return generalization metrics

#         raise NotImplementedError


# def load_trained_models(checkpoint_dir: str) -> Tuple[nn.Module, nn.Module]:
#     """
#     Load trained agent models from checkpoint.

#     Args:
#         checkpoint_dir: Directory containing saved models

#     Returns:
#         model_A: Agent A's trained model
#         model_B: Agent B's trained model
#     """
#     # TODO: Load model architectures
#     # TODO: Load trained weights
#     # TODO: Return initialized models

#     raise NotImplementedError


# def create_evaluation_report(results: Dict, save_path: str = 'results/evaluation_report.json') -> None:
#     """
#     Create comprehensive evaluation report.

#     Args:
#         results: Evaluation results
#         save_path: Path to save report
#     """
#     # TODO: Format results
#     # TODO: Add summary statistics
#     # TODO: Save as JSON report

#     raise NotImplementedError


# def main():
#     """
#     Run full evaluation suite on trained models.
#     """
#     # TODO: Load trained models
#     # TODO: Create environment
#     # TODO: Initialize evaluator
#     # TODO: Run performance evaluation
#     # TODO: Analyze communication
#     # TODO: Test generalization
#     # TODO: Create visualizations
#     # TODO: Generate report

#     raise NotImplementedError


# if __name__ == '__main__':
#     main()



"""
Evaluation script for trained multi-agent models.
Analyzes performance across the three ablation study modes.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Tuple, List, Dict
from multi_agent_env import MultiAgentEnv
from models import AgentDQN
from train import apply_observation_mask # Re-use the masking logic


# --- Evaluator Class ---

class MultiAgentEvaluator:
    """
    Evaluator for analyzing trained multi-agent policies.
    """

    def __init__(self, env: MultiAgentEnv, model_A: nn.Module, model_B: nn.Module, mode: str):
        """
        Initialize evaluator.

        Args:
            env: Multi-agent environment
            model_A: Trained model for Agent A
            model_B: Trained model for Agent B
            mode: The ablation mode being evaluated ('independent', 'comm', 'full')
        """
        self.env = env
        self.model_A = model_A
        self.model_B = model_B
        self.mode = mode
        self.device = torch.device("cpu")

        # Move models to device and set to evaluation mode
        self.model_A.to(self.device).eval()
        self.model_B.to(self.device).eval()
        
        # Ensure models are correctly recognized as AgentDQN for type hinting if not using TorchScript
        # Since we use TorchScript, we rely on its internal graph structure.

    @torch.no_grad()
    def run_episode(self) -> Dict:
        """
        Run single evaluation episode using the greedy policy.

        Returns:
            info: Episode statistics
        """
        obs_A, obs_B = self.env.reset()
        episode_reward = 0.0
        step_count = 0
        done = False
        
        # Tracking variables
        comm_A_history = []
        comm_B_history = []
        
        while not done:
            # 1. Apply masking based on the scenario mode
            masked_A = apply_observation_mask(obs_A, self.mode)
            masked_B = apply_observation_mask(obs_B, self.mode)
            
            # 2. Convert to tensors
            state_A_t = torch.tensor(masked_A, dtype=torch.float32).unsqueeze(0).to(self.device)
            state_B_t = torch.tensor(masked_B, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 3. Get Q-values and communication signal (Greedy Policy)
            qA, comm_A_t = self.model_A(state_A_t)
            qB, comm_B_t = self.model_B(state_B_t)
            
            # 4. Select greedy action
            action_A = qA.argmax(dim=1).item()
            action_B = qB.argmax(dim=1).item()
            
            # Get communication signal
            comm_A = comm_A_t.squeeze(0).item()
            comm_B = comm_B_t.squeeze(0).item()
            
            comm_A_history.append(comm_A)
            comm_B_history.append(comm_B)

            # 5. Execute step
            (obs_A, obs_B), reward, done = self.env.step(action_A, action_B, comm_A, comm_B)
            
            episode_reward += reward
            step_count += 1
        
        # Check final success
        # success = (obs_A == self.env.target_pos and obs_B == self.env.target_pos)

        pos_A = self.env.agent_positions[0]
        pos_B = self.env.agent_positions[1]
        success = (pos_A == self.env.target_pos and pos_B == self.env.target_pos)

        info = {
            'reward': episode_reward,
            'success': success,
            'steps': step_count,
            'comm_A_mean': np.mean(comm_A_history) if comm_A_history else 0.0,
            'comm_B_mean': np.mean(comm_B_history) if comm_B_history else 0.0,
            'comm_A_std': np.std(comm_A_history) if comm_A_history else 0.0,
            'comm_B_std': np.std(comm_B_history) if comm_B_history else 0.0,
        }
        return info

    def evaluate_performance(self, num_episodes: int = 100) -> Dict:
        """
        Evaluate overall performance statistics.
        """
        all_results = [self.run_episode() for _ in range(num_episodes)]
        
        rewards = [r['reward'] for r in all_results]
        successes = [r['success'] for r in all_results]
        steps = [r['steps'] for r in all_results if r['success']] # Only count steps for successful runs
        
        success_rate = np.mean(successes)
        
        stats = {
            'mode': self.mode,
            'num_episodes': num_episodes,
            'mean_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'success_rate': float(success_rate),
            'mean_steps_on_success': float(np.mean(steps)) if steps else self.env.max_steps,
            'median_steps_on_success': float(np.median(steps)) if steps else self.env.max_steps,
            'comm_A_mean_over_all': float(np.mean([r['comm_A_mean'] for r in all_results])),
            'comm_B_mean_over_all': float(np.mean([r['comm_B_mean'] for r in all_results])),
        }
        return stats
    
    # Placeholder for advanced analysis functions (as they require visualization libraries)
    def analyze_communication(self) -> Dict:
        """Analyze emergent communication protocols (placeholder)."""
        return {"analysis_note": "Communication analysis requires detailed tracking beyond simple mean/std."}
    
    def visualize_trajectory(self) -> None:
        """Visualize agent trajectories (placeholder)."""
        print("Trajectory visualization not implemented.")
        pass

# --- Utility Functions ---

def load_trained_models(checkpoint_dir: str) -> Tuple[nn.Module, nn.Module]:
    """
    Load trained agent models from TorchScript checkpoint.

    Args:
        checkpoint_dir: Directory containing saved 'agentA_final.pt' and 'agentB_final.pt'.

    Returns:
        model_A: Agent A's trained model (TorchScript traced model)
        model_B: Agent B's trained model (TorchScript traced model)
    """
    path_A = os.path.join(checkpoint_dir, "agentA_final.pt")
    path_B = os.path.join(checkpoint_dir, "agentB_final.pt")

    if not os.path.exists(path_A) or not os.path.exists(path_B):
        raise FileNotFoundError(f"Missing model files in {checkpoint_dir}. Ensure training finished successfully.")
    
    # Load TorchScript models
    model_A = torch.jit.load(path_A)
    model_B = torch.jit.load(path_B)
    
    # Set to evaluation mode (though TorchScript loaded models are often default eval)
    model_A.eval()
    model_B.eval()
    
    return model_A, model_B


def create_evaluation_report(results: Dict, save_path: str) -> None:
    """
    Create comprehensive evaluation report and save as JSON.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nEvaluation report saved to {save_path}")


def main():
    """
    Run full evaluation suite on trained models for all three modes.
    """
    # Define modes and directories
    modes = ['independent', 'comm', 'full']
    base_model_dir = os.path.join("problem2", "results", "agent_models")
    report_path = os.path.join("problem2", "results", "evaluation_results", "ablation_report.json")
    
    # Environment parameters (must match training parameters)
    env = MultiAgentEnv(grid_size=(10, 10), max_steps=50, seed=641)
    
    all_ablation_results = {}

    print("--- Starting Multi-Agent Ablation Evaluation ---")
    
    for mode in modes:
        print(f"\n[Evaluating Mode: {mode.upper()}]")
        try:
            # 1. Load trained models
            checkpoint_dir = os.path.join(base_model_dir, mode)
            model_A, model_B = load_trained_models(checkpoint_dir)
            
            # 2. Initialize evaluator
            evaluator = MultiAgentEvaluator(env, model_A, model_B, mode)
            
            # 3. Run performance evaluation (100 episodes)
            performance_stats = evaluator.evaluate_performance(num_episodes=100)
            
            all_ablation_results[mode] = performance_stats
            
            # Print summary for current mode
            print(f"  Success Rate: {performance_stats['success_rate']:.3f}")
            print(f"  Avg Reward: {performance_stats['mean_reward']:.2f}")
            print(f"  Avg Steps (Success): {performance_stats['mean_steps_on_success']:.1f}")
            
        except FileNotFoundError as e:
            print(f"  ERROR: Skipping mode '{mode}' - {e}")
            all_ablation_results[mode] = {"error": str(e), "note": "Model files not found. Run train.py first."}
        except Exception as e:
            print(f"  An unexpected error occurred during evaluation for mode '{mode}': {e}")
            all_ablation_results[mode] = {"error": str(e)}

    # 4. Generate final report
    create_evaluation_report(all_ablation_results, report_path)
    print("--- Evaluation Complete ---")


if __name__ == '__main__':
    main()