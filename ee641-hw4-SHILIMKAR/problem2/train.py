# """
# Training script for multi-agent DQN with communication.
# """

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import numpy as np
# import argparse
# import json
# import os
# from typing import Tuple, Optional
# from multi_agent_env import MultiAgentEnv
# from models import AgentDQN
# from replay_buffer import ReplayBuffer


# def apply_observation_mask(obs: np.ndarray, mode: str) -> np.ndarray:
#     """
#     Apply masking to observation based on ablation mode.

#     Args:
#         obs: 11-dimensional observation vector
#         mode: One of 'independent', 'comm', 'full'

#     Returns:
#         Masked observation
#     """
#     # TODO: Implement masking logic
#     # 'independent': Set elements 9 and 10 to zero
#     # 'comm': Set element 10 to zero
#     # 'full': No masking

#     raise NotImplementedError


# class MultiAgentTrainer:
#     """
#     Trainer for multi-agent DQN system.

#     Handles training loop, exploration, and network updates.
#     """

#     def __init__(self, env: MultiAgentEnv, args):
#         """
#         Initialize trainer.

#         Args:
#             env: Multi-agent environment
#             args: Training arguments
#         """
#         self.env = env
#         self.args = args

#         # Use CPU for small networks
#         self.device = torch.device("cpu")

#         # TODO: Initialize networks for both agents (remember to .to(self.device))
#         # TODO: Initialize target networks (if using)
#         # TODO: Initialize optimizers
#         # TODO: Initialize replay buffer
#         # TODO: Initialize epsilon for exploration

#         raise NotImplementedError

#     def select_action(self, state: np.ndarray, network: nn.Module,
#                       epsilon: float) -> Tuple[int, float]:
#         """
#         Select action using epsilon-greedy policy.

#         Args:
#             state: Agent observation (11-dimensional, may need masking)
#             network: Agent's DQN
#             epsilon: Exploration probability

#         Returns:
#             action: Selected action
#             comm_signal: Communication signal
#         """
#         # TODO: Apply observation masking based on self.args.mode
#         #       masked_state = apply_observation_mask(state, self.args.mode)
#         # TODO: With probability epsilon, select random action
#         # TODO: Otherwise, select action with highest Q-value
#         # TODO: Always get communication signal from network
#         # TODO: Return (action, comm_signal)

#         raise NotImplementedError

#     def update_networks(self, batch_size: int) -> float:
#         """
#         Sample batch and update both agent networks.

#         Args:
#             batch_size: Size of training batch

#         Returns:
#             loss: Combined loss value
#         """
#         # TODO: Sample batch from replay buffer
#         # TODO: Convert to tensors and move to device
#         # TODO: Compute Q-values for current states
#         # TODO: Compute target Q-values using target networks
#         # TODO: Calculate TD loss for both agents
#         # TODO: Backpropagate and update networks
#         # TODO: Return combined loss

#         raise NotImplementedError

#     def train_episode(self) -> Tuple[float, bool]:
#         """
#         Run one training episode.

#         Returns:
#             episode_reward: Total reward for episode
#             success: Whether agents reached target
#         """
#         # TODO: Reset environment
#         # TODO: Initialize episode variables
#         # TODO: Run episode until termination:
#         #       - Select actions for both agents
#         #       - Execute actions in environment
#         #       - Store transition in replay buffer
#         #       - Update networks if enough samples
#         # TODO: Return episode reward and success flag

#         raise NotImplementedError

#     def train(self) -> None:
#         """
#         Main training loop.
#         """
#         # TODO: Create results directories
#         # TODO: Initialize logging
#         # TODO: Main training loop:
#         #       - Run episodes
#         #       - Update epsilon
#         #       - Update target networks periodically
#         #       - Log progress
#         #       - Save checkpoints
#         # TODO: Save final models including TorchScript format:
#         #       scripted_model = torch.jit.script(self.network_A)
#         #       scripted_model.save("dqn_net.pt")

#         raise NotImplementedError

#     def evaluate(self, num_episodes: int = 10) -> Tuple[float, float]:
#         """
#         Evaluate current policy.

#         Args:
#             num_episodes: Number of evaluation episodes

#         Returns:
#             mean_reward: Average reward
#             success_rate: Fraction of successful episodes
#         """
#         # TODO: Set networks to evaluation mode
#         # TODO: Run episodes without exploration
#         # TODO: Track rewards and successes
#         # TODO: Return statistics

#         raise NotImplementedError


# def main():
#     """
#     Parse arguments and run training.
#     """
#     parser = argparse.ArgumentParser(description='Train Multi-Agent DQN')

#     # Environment parameters
#     parser.add_argument('--grid_size', type=int, nargs=2, default=[10, 10],
#                        help='Grid dimensions')
#     parser.add_argument('--max_steps', type=int, default=50,
#                        help='Maximum steps per episode')

#     # Training parameters
#     parser.add_argument('--num_episodes', type=int, default=5000,
#                        help='Number of training episodes')
#     parser.add_argument('--batch_size', type=int, default=32,
#                        help='Batch size for training')
#     parser.add_argument('--lr', type=float, default=1e-3,
#                        help='Learning rate')
#     parser.add_argument('--gamma', type=float, default=0.99,
#                        help='Discount factor')

#     # Exploration parameters
#     parser.add_argument('--epsilon_start', type=float, default=1.0,
#                        help='Initial exploration rate')
#     parser.add_argument('--epsilon_end', type=float, default=0.05,
#                        help='Final exploration rate')
#     parser.add_argument('--epsilon_decay', type=float, default=0.995,
#                        help='Epsilon decay rate')

#     # Network parameters
#     parser.add_argument('--hidden_dim', type=int, default=64,
#                        help='Hidden layer size')
#     parser.add_argument('--target_update', type=int, default=100,
#                        help='Target network update frequency')

#     # Ablation study mode
#     parser.add_argument('--mode', type=str, default='full',
#                        choices=['independent', 'comm', 'full'],
#                        help='Information mode: independent (mask comm+dist), '
#                             'comm (mask dist only), full (no masking)')

#     # Other parameters
#     parser.add_argument('--seed', type=int, default=641,
#                        help='Random seed')
#     parser.add_argument('--save_freq', type=int, default=500,
#                        help='Model save frequency')

#     args = parser.parse_args()

#     # TODO: Set random seeds
#     # TODO: Create environment
#     # TODO: Create trainer
#     # TODO: Run training
#     # TODO: Final evaluation

#     raise NotImplementedError


# if __name__ == '__main__':
#     main()



"""
Training script for multi-agent DQN with communication.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
import random
from typing import Tuple, Optional, Dict
from multi_agent_env import MultiAgentEnv
from models import AgentDQN
from replay_buffer import ReplayBuffer

# --- Utility Functions ---

def apply_observation_mask(obs: np.ndarray, mode: str) -> np.ndarray:
    """
    Apply masking to observation based on ablation mode.

    Observation vector (11D):
    - Elements 0-8: Grid patch
    - Element 9: Communication scalar (c_in)
    - Element 10: Normalized L2 distance (dist)

    Args:
        obs: 11-dimensional observation vector
        mode: One of 'independent', 'comm', 'full'

    Returns:
        Masked observation (modified copy of input array)
    """
    masked_obs = np.copy(obs)
    
    if mode == 'independent':
        # (a) Independent: Mask elements 9-10 with zeros (Comm + Distance)
        masked_obs[9] = 0.0
        masked_obs[10] = 0.0
    elif mode == 'comm':
        # (b) Communication Only: Mask element 10 with zero (Distance only)
        # Element 9 (Comm) remains active.
        masked_obs[10] = 0.0
    elif mode == 'full':
        # (c) Full Information: No masking
        pass
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return masked_obs


def soft_update(target_net: nn.Module, policy_net: nn.Module, tau: float):
    """Soft update policy parameters toward target network."""
    for target_param, policy_param in zip(target_net.parameters(), policy_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


# --- Trainer Class ---

class MultiAgentTrainer:
    """
    Trainer for multi-agent DQN system.
    """

    def __init__(self, env: MultiAgentEnv, args):
        self.env = env
        self.args = args
        self.device = torch.device("cpu") # Keep it simple on CPU as specified

        # Network/RL parameters
        self.gamma = args.gamma
        self.target_update_freq = args.target_update
        self.obs_dim = 11 # Fixed 11D observation space
        self.action_dim = 5

        # 1. Initialize networks and target networks
        self.network_A = AgentDQN(input_dim=self.obs_dim, hidden_dim=args.hidden_dim, num_actions=self.action_dim).to(self.device)
        self.target_network_A = AgentDQN(input_dim=self.obs_dim, hidden_dim=args.hidden_dim, num_actions=self.action_dim).to(self.device)
        self.target_network_A.load_state_dict(self.network_A.state_dict())
        self.target_network_A.eval() # Set target network to evaluation mode

        # Since it's a homogeneous setup (same architecture, shared environment), 
        # we can use the same model class for Agent B, initialized separately.
        self.network_B = AgentDQN(input_dim=self.obs_dim, hidden_dim=args.hidden_dim, num_actions=self.action_dim).to(self.device)
        self.target_network_B = AgentDQN(input_dim=self.obs_dim, hidden_dim=args.hidden_dim, num_actions=self.action_dim).to(self.device)
        self.target_network_B.load_state_dict(self.network_B.state_dict())
        self.target_network_B.eval()
        
        # 2. Initialize optimizers
        self.optimizer_A = optim.Adam(self.network_A.parameters(), lr=args.lr)
        self.optimizer_B = optim.Adam(self.network_B.parameters(), lr=args.lr)

        # 3. Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=100000, seed=args.seed)

        # 4. Initialize epsilon for exploration
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_end = args.epsilon_end
        
        # Logging
        self.log = {'episode': [], 'reward': [], 'success_rate': [], 'loss': []}

    @torch.no_grad()
    def select_action(self, state: np.ndarray, network: nn.Module,
                      epsilon: float) -> Tuple[int, float]:
        """
        Select action using epsilon-greedy policy.
        """
        # 1. Apply observation masking
        masked_state = apply_observation_mask(state, self.args.mode)
        
        # Convert state to tensor for network input
        state_t = torch.tensor(masked_state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # 2. Get Q-values and communication signal from network
        network.eval()
        action_values, comm_signal_t = network(state_t)
        network.train()
        
        comm_signal = comm_signal_t.squeeze(0).item() # scalar communication output
        
        # 3. Epsilon-greedy exploration
        if random.random() < epsilon:
            # Explore: Select random movement action
            action = random.randrange(self.action_dim) 
        else:
            # Exploit: Select action with highest Q-value
            action = action_values.argmax(dim=1).item()
        
        return action, comm_signal

    def update_networks(self, batch_size: int) -> float:
        """
        Sample batch and update both agent networks.
        We implement **Decentralized Training with Centralized Reward/Experience** (DRL in a MARL setting).
        """
        if len(self.replay_buffer) < batch_size:
            return 0.0 # Not enough samples yet

        # Sample batch
        (sA, sB, aA, aB, cA, cB, r, sA_prime, sB_prime, done) = self.replay_buffer.sample(batch_size)
        
        # Apply masking to next states (sA_prime, sB_prime) before computing target Q-values
        # Note: Input masking is a static part of the observation generation for a given scenario.
        # We need to apply this mask to the sampled observations before feeding them to the network.
        sA_prime_masked = torch.tensor(np.stack([apply_observation_mask(s.numpy(), self.args.mode) for s in sA_prime]), dtype=torch.float32).to(self.device)
        sB_prime_masked = torch.tensor(np.stack([apply_observation_mask(s.numpy(), self.args.mode) for s in sB_prime]), dtype=torch.float32).to(self.device)
        
        # Apply masking to current states (sA, sB) for Q-value calculation
        sA_masked = torch.tensor(np.stack([apply_observation_mask(s.numpy(), self.args.mode) for s in sA]), dtype=torch.float32).to(self.device)
        sB_masked = torch.tensor(np.stack([apply_observation_mask(s.numpy(), self.args.mode) for s in sB]), dtype=torch.float32).to(self.device)

        # --- Compute Target Q-values (Y_j) ---
        
        # Calculate max Q' for A and B using TARGET networks (Double DQN not specified, so we use max Q)
        with torch.no_grad():
            # Target Q-values for next state s'
            qA_prime_target, _ = self.target_network_A(sA_prime_masked)
            qB_prime_target, _ = self.target_network_B(sB_prime_masked)

            # Max Q-value for next state
            max_qA_prime = qA_prime_target.max(1)[0].unsqueeze(1)
            max_qB_prime = qB_prime_target.max(1)[0].unsqueeze(1)
            
            # Joint Target Q-value: Since the reward is shared and termination is joint,
            # the target is based on the joint state-action value approximation.
            # In decentralized Q-learning for shared reward, the Bellman update is:
            # Y_j = r + gamma * max_{aA', aB'} Q(s', aA', aB')
            # Assuming additivity/factorization for simplicity (as common in independent Q-learning/MADDPG):
            # max Q(s', aA', aB') approx max_aA' Q_A(s', aA') + max_aB' Q_B(s', aB') 
            # *However, in MARL with a shared reward, it's often more effective to sum the max Q's as an approximation
            # of the joint value function:*
            
            # Y_j = r + gamma * (1 - done) * (max_qA_prime + max_qB_prime) / 2
            # For simplicity and standard MADQN/IQL, we use the shared reward directly:
            max_joint_q_prime = (max_qA_prime + max_qB_prime) / 2.0
            
            y_j = r + self.gamma * (1 - done) * max_joint_q_prime 

        # --- Compute Current Q-values (Q(s, a)) ---

        # 1. Agent A
        qA_current, _ = self.network_A(sA_masked)
        # Q-value for the action that was actually taken
        qA_s_a = qA_current.gather(1, aA)

        # 2. Agent B
        qB_current, _ = self.network_B(sB_masked)
        # Q-value for the action that was actually taken
        qB_s_a = qB_current.gather(1, aB)
        
        # Joint Current Q-value (for loss calculation)
        current_joint_q = (qA_s_a + qB_s_a) / 2.0
        
        # --- Compute Loss ---

        # TD Loss (Mean Squared Error) for the joint Q-value estimation
        loss = F.mse_loss(current_joint_q, y_j)

        # Backpropagation and optimization
        self.optimizer_A.zero_grad()
        self.optimizer_B.zero_grad()
        
        # Since the loss is a function of both networks, we can backpropagate through both.
        loss.backward()
        
        # Gradient clipping (optional but good practice)
        torch.nn.utils.clip_grad_norm_(self.network_A.parameters(), max_norm=10.0)
        torch.nn.utils.clip_grad_norm_(self.network_B.parameters(), max_norm=10.0)
        
        self.optimizer_A.step()
        self.optimizer_B.step()

        return loss.item()

    def train_episode(self, episode_num: int) -> Tuple[float, bool]:
        """
        Run one training episode.
        """
        obs_A, obs_B = self.env.reset()
        episode_reward = 0.0
        done = False
        
        # We need to track the communication signals used for the environment step
        # They come from the *previous* observation's comm_in component, but the 
        # problem is implicitly asking for the *output* comm signal from the network.
        comm_A_out, comm_B_out = 0.0, 0.0 # Initial communication signals for the *first* step

        while not done:
            # 1. Select actions and get new comm signals for the step
            action_A, comm_A_out = self.select_action(obs_A, self.network_A, self.epsilon)
            action_B, comm_B_out = self.select_action(obs_B, self.network_B, self.epsilon)
            
            # Store current states before the step
            sA, sB = obs_A, obs_B 
            
            # 2. Execute step
            (obs_A_prime, obs_B_prime), reward, done = self.env.step(action_A, action_B, comm_A_out, comm_B_out)
            
            # 3. Store transition: (s, a, c, r, s', done)
            # The comm signals stored are the *output* cA, cB that were used in the step.
            self.replay_buffer.push(
                sA, sB, action_A, action_B, comm_A_out, comm_B_out, reward, 
                obs_A_prime, obs_B_prime, done
            )
            
            # Update state for next iteration
            obs_A, obs_B = obs_A_prime, obs_B_prime
            episode_reward += reward

            # 4. Update networks if buffer is large enough
            if len(self.replay_buffer) > self.args.batch_size * 5: # Start training after a small buffer fill
                loss = self.update_networks(self.args.batch_size)
                if loss > 0:
                     self.log['loss'].append(loss)

            # 5. Update target networks periodically
            if episode_num % self.target_update_freq == 0 and self.env.step_count % self.target_update_freq == 0:
                # Use hard update (load_state_dict) as soft update (tau<1) is not specified.
                self.target_network_A.load_state_dict(self.network_A.state_dict())
                self.target_network_B.load_state_dict(self.network_B.state_dict())
                
        # Update epsilon after episode end
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # success = (obs_A == self.env.target_pos and obs_B == self.env.target_pos)
        # return episode_reward, success
        pos_A = self.env.agent_positions[0]
        pos_B = self.env.agent_positions[1]
        
        # Check if both final positions match the target position
        success = (pos_A == self.env.target_pos and pos_B == self.env.target_pos)
        return episode_reward, success


    @torch.no_grad()
    def evaluate(self, num_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate current policy (greedy actions).
        """
        self.network_A.eval()
        self.network_B.eval()
        
        total_reward = 0.0
        total_success = 0
        
        for _ in range(num_episodes):
            obs_A, obs_B = self.env.reset()
            done = False
            
            while not done:
                # 1. Apply masking to observation
                masked_A = apply_observation_mask(obs_A, self.args.mode)
                masked_B = apply_observation_mask(obs_B, self.args.mode)
                
                # Convert to tensors
                state_A_t = torch.tensor(masked_A, dtype=torch.float32).unsqueeze(0).to(self.device)
                state_B_t = torch.tensor(masked_B, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 2. Get Q-values and communication signal
                qA, comm_A_t = self.network_A(state_A_t)
                qB, comm_B_t = self.network_B(state_B_t)
                
                # 3. Select greedy action
                action_A = qA.argmax(dim=1).item()
                action_B = qB.argmax(dim=1).item()
                
                # Get communication signal
                comm_A = comm_A_t.squeeze(0).item()
                comm_B = comm_B_t.squeeze(0).item()

                # 4. Execute step
                # (obs_A, obs_B), reward, done = self.env.step(action_A, action_B, comm_A, comm_B)
                # total_reward += reward

                # if done and (obs_A == self.env.target_pos and obs_B == self.env.target_pos):
                #     total_success += 1
                (obs_A, obs_B), reward, done = self.env.step(action_A, action_B, comm_A, comm_B)
                total_reward += reward

                # Retrieve the final positions from the environment's state
                pos_A = self.env.agent_positions[0]
                pos_B = self.env.agent_positions[1]
                
                # Check for success using the actual positions
                if done and (pos_A == self.env.target_pos and pos_B == self.env.target_pos):
                    total_success += 1
        
        self.network_A.train()
        self.network_B.train()
        
        mean_reward = total_reward / num_episodes
        success_rate = total_success / num_episodes
        
        return mean_reward, success_rate

    def train(self) -> None:
        """
        Main training loop.
        """
        # Create results directory
        model_dir = os.path.join("problem2", "results", "agent_models", self.args.mode)
        log_path = os.path.join("problem2", "results", "training_logs", f"{self.args.mode}_log.json")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        print(f"\n--- Starting Training for Mode: {self.args.mode.upper()} ---")
        print(f"Total Episodes: {self.args.num_episodes}")

        for episode in range(1, self.args.num_episodes + 1):
            episode_reward, success = self.train_episode(episode)
            
            # Log progress
            self.log['episode'].append(episode)
            self.log['reward'].append(episode_reward)
            self.log['success_rate'].append(float(success))
            
            if episode % 100 == 0:
                mean_reward, success_rate = self.evaluate(num_episodes=10)
                print(f"Eps {episode}/{self.args.num_episodes} | Epsilon: {self.epsilon:.4f} | Avg R: {mean_reward:.2f} | Success Rate: {success_rate:.2f}")

                # Save log
                with open(log_path, 'w') as f:
                    json.dump(self.log, f, indent=4)
            
            # Save checkpoints
            if episode % self.args.save_freq == 0 or episode == self.args.num_episodes:
                torch.save(self.network_A.state_dict(), os.path.join(model_dir, f"A_{episode}.pth"))
                torch.save(self.network_B.state_dict(), os.path.join(model_dir, f"B_{episode}.pth"))

        # Save final models in TorchScript format
        print(f"\nTraining finished. Saving final models to {model_dir}")
        scripted_model_A = torch.jit.script(self.network_A)
        scripted_model_B = torch.jit.script(self.network_B)
        
        scripted_model_A.save(os.path.join(model_dir, "agentA_final.pt"))
        scripted_model_B.save(os.path.join(model_dir, "agentB_final.pt"))

        # Final Evaluation
        final_mean_reward, final_success_rate = self.evaluate(num_episodes=100)
        print(f"\nFinal Evaluation (100 episodes): Mean Reward: {final_mean_reward:.2f}, Success Rate: {final_success_rate:.2f}")


def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # If using CUDA: torch.backends.cudnn.deterministic = True
    #                 torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train Multi-Agent DQN')

    # Environment parameters
    parser.add_argument('--grid_size', type=int, nargs=2, default=[10, 10], help='Grid dimensions')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per episode')

    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=5000, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')

    # Exploration parameters
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.05, help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon decay rate')

    # Network parameters
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden layer size')
    parser.add_argument('--target_update', type=int, default=100, help='Target network update frequency')

    # Ablation study mode
    # To run the full ablation, the user must run the script three times, changing this parameter.
    parser.add_argument('--mode', type=str, default='full',
                       choices=['independent', 'comm', 'full'],
                       help='Information mode: independent (mask comm+dist), '
                            'comm (mask dist only), full (no masking)')

    # Other parameters
    parser.add_argument('--seed', type=int, default=641, help='Random seed')
    parser.add_argument('--save_freq', type=int, default=500, help='Model save frequency')

    args = parser.parse_args()

    # Set random seeds
    set_seeds(args.seed)

    # Create environment
    env = MultiAgentEnv(grid_size=tuple(args.grid_size), max_steps=args.max_steps, seed=args.seed)

    # Create trainer
    trainer = MultiAgentTrainer(env, args)

    # Run training
    trainer.train()

    # Note: Final evaluation is included in trainer.train()

if __name__ == '__main__':
    main()