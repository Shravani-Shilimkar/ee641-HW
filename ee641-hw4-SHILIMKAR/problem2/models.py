# """
# Neural network models for multi-agent DQN with communication.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from typing import Tuple


# class AgentDQN(nn.Module):
#     """
#     Deep Q-Network for agent with communication capability.

#     Network processes observations and outputs both Q-values and communication signal.
#     """

#     def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_actions: int = 5):
#         """
#         Initialize DQN with dual outputs.

#         Args:
#             input_dim: Dimension of input observation (default 10)
#             hidden_dim: Number of hidden units
#             num_actions: Number of discrete actions (default 5)
#         """
#         super(AgentDQN, self).__init__()

#         # TODO: Define network layers
#         #       - Input layer: input_dim -> hidden_dim
#         #       - Hidden layers (at least one more)
#         #       - Action head: outputs Q-values for each action
#         #       - Communication head: outputs single scalar

#         raise NotImplementedError

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward pass through network.

#         Args:
#             x: Input tensor of shape [batch_size, input_dim]

#         Returns:
#             action_values: Q-values for each action [batch_size, num_actions]
#             comm_signal: Communication signal in [0,1] [batch_size, 1]
#         """
#         # TODO: Pass input through shared layers
#         # TODO: Compute action Q-values through action head
#         # TODO: Compute communication signal through comm head
#         # TODO: Apply sigmoid to bound communication in [0,1]
#         # TODO: Return (action_values, comm_signal)

#         raise NotImplementedError


# class DuelingDQN(nn.Module):
#     """
#     Dueling DQN architecture for improved value estimation.

#     Separates value and advantage streams for better learning.
#     """

#     def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_actions: int = 5):
#         """
#         Initialize Dueling DQN.

#         Args:
#             input_dim: Dimension of input observation
#             hidden_dim: Number of hidden units
#             num_actions: Number of discrete actions
#         """
#         super(DuelingDQN, self).__init__()

#         # TODO: Define shared feature layers
#         # TODO: Define value stream (outputs single value)
#         # TODO: Define advantage stream (outputs advantages for each action)
#         # TODO: Define communication head

#         raise NotImplementedError

#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward pass with dueling architecture.

#         Args:
#             x: Input tensor [batch_size, input_dim]

#         Returns:
#             q_values: Combined Q-values [batch_size, num_actions]
#             comm_signal: Communication signal in [0,1] [batch_size, 1]
#         """
#         # TODO: Compute shared features
#         # TODO: Compute state value V(s)
#         # TODO: Compute advantages A(s,a)
#         # TODO: Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
#         # TODO: Compute communication signal
#         # TODO: Apply sigmoid to bound communication in [0,1]
#         # TODO: Return (q_values, comm_signal)

#         raise NotImplementedError



"""
Neural network models for multi-agent DQN with communication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class AgentDQN(nn.Module):
    """
    Deep Q-Network for agent with communication capability.

    Network processes observations (11D) and outputs both Q-values (5D) and
    communication signal (1D) following the architecture:
    h = ReLU(W1 * x + b1)
    Q(s, a) = W_action * h + b_action
    c_out = sigma(W_comm * h + b_comm)
    """

    # Note: Updated default input_dim to 11 to match the environment observation size
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, num_actions: int = 5):
        """
        Initialize DQN with dual outputs.

        Args:
            input_dim: Dimension of input observation (11D: Grid 0-8, Comm 9, Dist 10)
            hidden_dim: Number of hidden units
            num_actions: Number of discrete actions (5: Up, Down, Left, Right, Stay)
        """
        super(AgentDQN, self).__init__()

        self.num_actions = num_actions
        self.input_dim = input_dim

        # 1. Shared Feature Layer (W1)
        # Input: 11D observation vector (x)
        # Output: hidden_dim (h)
        self.fc_shared = nn.Linear(input_dim, hidden_dim)

        # 2. Action Head (W_action)
        # Input: hidden_dim (h)
        # Output: num_actions (Q-values)
        self.fc_action = nn.Linear(hidden_dim, num_actions)

        # 3. Communication Head (W_comm)
        # Input: hidden_dim (h)
        # Output: 1 (Communication scalar c_out)
        self.fc_comm = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through network.

        Args:
            x: Input tensor of shape [batch_size, input_dim]

        Returns:
            action_values: Q-values for each action [batch_size, num_actions]
            comm_signal: Communication signal in [0,1] [batch_size, 1]
        """
        # 1. Shared Feature (h) with ReLU activation
        h = F.relu(self.fc_shared(x))

        # 2. Compute Action Q-values
        action_values = self.fc_action(h) # Output size: [batch_size, num_actions]

        # 3. Compute Communication Signal
        # Apply sigmoid to bound c_out in [0, 1] as per the equation: c_out = sigma(...)
        comm_signal_logits = self.fc_comm(h)
        comm_signal = torch.sigmoid(comm_signal_logits) # Output size: [batch_size, 1]

        return action_values, comm_signal


class DuelingDQN(nn.Module):
    # This class is left unimplemented as the core problem focuses on the AgentDQN 
    # and the provided image specified a standard Q-learning architecture with a shared hidden layer.
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64, num_actions: int = 5):
        super(DuelingDQN, self).__init__()
        raise NotImplementedError("DuelingDQN is not required by the core specification and is not implemented.")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError