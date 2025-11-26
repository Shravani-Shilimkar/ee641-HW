# """
# Experience replay buffer for multi-agent DQN training.
# """

# import numpy as np
# import random
# from typing import Tuple, List, Optional
# from collections import deque


# class ReplayBuffer:
#     """
#     Experience replay buffer for storing and sampling transitions.

#     Stores joint experiences from both agents for coordinated learning.
#     """

#     def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
#         """
#         Initialize replay buffer.

#         Args:
#             capacity: Maximum number of transitions to store
#             seed: Random seed for sampling
#         """
#         self.capacity = capacity
#         self.buffer = deque(maxlen=capacity)

#         if seed is not None:
#             random.seed(seed)
#             np.random.seed(seed)

#     def push(self, state_A: np.ndarray, state_B: np.ndarray,
#              action_A: int, action_B: int,
#              comm_A: float, comm_B: float,
#              reward: float,
#              next_state_A: np.ndarray, next_state_B: np.ndarray,
#              done: bool) -> None:
#         """
#         Store a transition in the buffer.

#         Args:
#             state_A: Agent A's observation
#             state_B: Agent B's observation
#             action_A: Agent A's action
#             action_B: Agent B's action
#             comm_A: Communication from A to B
#             comm_B: Communication from B to A
#             reward: Shared reward
#             next_state_A: Agent A's next observation
#             next_state_B: Agent B's next observation
#             done: Whether episode terminated
#         """
#         # TODO: Create transition tuple
#         # TODO: Add to buffer (automatic removal of oldest if at capacity)

#         raise NotImplementedError

#     def sample(self, batch_size: int) -> Tuple:
#         """
#         Sample a batch of transitions.

#         Args:
#             batch_size: Number of transitions to sample

#         Returns:
#             Batch of transitions as separate arrays for each component
#         """
#         # TODO: Sample batch_size transitions randomly
#         # TODO: Separate components into individual arrays
#         # TODO: Convert to appropriate numpy arrays
#         # TODO: Return tuple of arrays

#         raise NotImplementedError

#     def __len__(self) -> int:
#         """
#         Get current size of buffer.

#         Returns:
#             Number of transitions in buffer
#         """
#         return len(self.buffer)


# class PrioritizedReplayBuffer:
#     """
#     Prioritized experience replay for importance sampling.

#     Samples transitions based on TD-error magnitude.
#     """

#     def __init__(self, capacity: int = 10000, alpha: float = 0.6,
#                  beta_start: float = 0.4, beta_steps: int = 100000,
#                  seed: Optional[int] = None):
#         """
#         Initialize prioritized replay buffer.

#         Args:
#             capacity: Maximum number of transitions
#             alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
#             beta_start: Initial importance sampling weight
#             beta_steps: Steps to anneal beta to 1.0
#             seed: Random seed
#         """
#         self.capacity = capacity
#         self.alpha = alpha
#         self.beta = beta_start
#         self.beta_start = beta_start
#         self.beta_steps = beta_steps
#         self.frame = 1

#         # TODO: Initialize data storage
#         # TODO: Initialize priority tree (sum-tree or similar)
#         # TODO: Set random seed if provided

#         raise NotImplementedError

#     def push(self, *args, **kwargs) -> None:
#         """
#         Store transition with maximum priority.

#         New transitions get maximum priority to ensure they're sampled at least once.
#         """
#         # TODO: Store transition
#         # TODO: Assign maximum priority to new transition

#         raise NotImplementedError

#     def sample(self, batch_size: int) -> Tuple:
#         """
#         Sample batch with prioritization.

#         Returns:
#             transitions: Batch of transitions
#             weights: Importance sampling weights
#             indices: Indices for updating priorities
#         """
#         # TODO: Update beta based on schedule
#         # TODO: Sample transitions based on priorities
#         # TODO: Calculate importance sampling weights
#         # TODO: Return transitions, weights, and indices

#         raise NotImplementedError

#     def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
#         """
#         Update priorities for sampled transitions.

#         Args:
#             indices: Indices of transitions to update
#             priorities: New priority values (typically TD-errors)
#         """
#         # TODO: Update priorities for given indices
#         # TODO: Apply alpha exponent for prioritization

#         raise NotImplementedError



"""
Experience replay buffer for multi-agent DQN training.
"""

import numpy as np
import random
import torch
from typing import Tuple, List, Optional
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.

    Stores joint experiences from both agents for coordinated learning.
    """

    # Define the structure of the transition tuple for clarity
    Transition = Tuple[
        np.ndarray, np.ndarray, # s_A, s_B
        int, int,               # a_A, a_B
        float, float,           # c_A, c_B
        float,                  # r
        np.ndarray, np.ndarray, # s'_A, s'_B
        bool                    # done
    ]

    def __init__(self, capacity: int = 10000, seed: Optional[int] = None):
        """
        Initialize replay buffer.
        """
        self.capacity = capacity
        # The deque handles the capacity limit automatically (removes oldest when full)
        self.buffer = deque(maxlen=capacity)

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self._rng = random

    def push(self, state_A: np.ndarray, state_B: np.ndarray,
             action_A: int, action_B: int,
             comm_A: float, comm_B: float,
             reward: float,
             next_state_A: np.ndarray, next_state_B: np.ndarray,
             done: bool) -> None:
        """
        Store a transition in the buffer.
        """
        # Create the transition tuple
        transition: ReplayBuffer.Transition = (
            state_A, state_B, 
            action_A, action_B, 
            comm_A, comm_B, 
            reward, 
            next_state_A, next_state_B, 
            done
        )
        # Add to buffer
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions.

        Returns:
            Tuple of torch.Tensors for each component, ready for network input.
        """
        # Ensure we don't sample more than available transitions
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        # Separate components into individual lists
        states_A, states_B, \
        actions_A, actions_B, \
        comms_A, comms_B, \
        rewards, \
        next_states_A, next_states_B, \
        dones = zip(*batch)

        # Convert to numpy arrays, then to torch tensors
        # Observations: Batch_size x Obs_dim (e.g., 11)
        # Actions: Batch_size x 1 (int)
        # Scalars (Comm, Reward, Done): Batch_size x 1

        state_A_t = torch.tensor(np.array(states_A), dtype=torch.float32)
        state_B_t = torch.tensor(np.array(states_B), dtype=torch.float32)

        action_A_t = torch.tensor(np.array(actions_A), dtype=torch.int64).unsqueeze(-1)
        action_B_t = torch.tensor(np.array(actions_B), dtype=torch.int64).unsqueeze(-1)
        
        comm_A_t = torch.tensor(np.array(comms_A), dtype=torch.float32).unsqueeze(-1)
        comm_B_t = torch.tensor(np.array(comms_B), dtype=torch.float32).unsqueeze(-1)

        reward_t = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(-1)

        next_state_A_t = torch.tensor(np.array(next_states_A), dtype=torch.float32)
        next_state_B_t = torch.tensor(np.array(next_states_B), dtype=torch.float32)

        # Convert boolean 'done' to float tensor (0.0 or 1.0)
        done_t = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(-1)

        return (state_A_t, state_B_t, action_A_t, action_B_t, 
                comm_A_t, comm_B_t, reward_t, 
                next_state_A_t, next_state_B_t, done_t)

    def __len__(self) -> int:
        """
        Get current size of buffer.
        """
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay for importance sampling.
    (Not required for the base problem, implementation skipped.)
    """

    def __init__(self, capacity: int = 10000, alpha: float = 0.6,
                 beta_start: float = 0.4, beta_steps: int = 100000,
                 seed: Optional[int] = None):
        """
        Initialize prioritized replay buffer.
        """
        raise NotImplementedError("PrioritizedReplayBuffer is not implemented as it is not strictly required.")

    def push(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def sample(self, batch_size: int) -> Tuple:
        raise NotImplementedError

    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        raise NotImplementedError