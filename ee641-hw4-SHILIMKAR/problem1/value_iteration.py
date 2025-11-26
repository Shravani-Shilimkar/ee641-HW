# """
# Value Iteration algorithm for solving MDPs.
# """

# import numpy as np
# from typing import Tuple, Optional
# from environment import GridWorldEnv


# class ValueIteration:
#     """
#     Value Iteration solver for gridworld MDP.

#     Computes optimal value function V* using dynamic programming.
#     """

#     def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
#         """
#         Initialize Value Iteration solver.

#         Args:
#             env: GridWorld environment
#             gamma: Discount factor
#             epsilon: Convergence threshold
#         """
#         self.env = env
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.n_states = env.grid_size ** 2
#         self.n_actions = env.action_space

#     def solve(self, max_iterations: int = 1000) -> Tuple[np.ndarray, int]:
#         """
#         Run value iteration until convergence.

#         Args:
#             max_iterations: Maximum number of iterations

#         Returns:
#             values: Converged value function V(s)
#             n_iterations: Number of iterations until convergence
#         """
#         # TODO: Initialize value function to zeros
#         # TODO: Iterate until convergence:
#         #       - For each state:
#         #           - Compute Q(s,a) for all actions using Bellman backup
#         #           - Set V(s) = max_a Q(s,a)
#         #       - Check convergence: max|V_new - V_old| < epsilon
#         #       - Update value function
#         # TODO: Return final values and iteration count

#         raise NotImplementedError

#     def compute_q_values(self, state: int, values: np.ndarray) -> np.ndarray:
#         """
#         Compute Q-values for all actions in a state.

#         Args:
#             state: State index
#             values: Current value function

#         Returns:
#             q_values: Array of Q(s,a) for each action
#         """
#         # TODO: For each action:
#         #       - Get transition probabilities P(s'|s,a)
#         #       - Compute expected value:
#         #           Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
#         # TODO: Return Q-values array

#         raise NotImplementedError

#     def extract_policy(self, values: np.ndarray) -> np.ndarray:
#         """
#         Extract optimal policy from value function.

#         Args:
#             values: Optimal value function

#         Returns:
#             policy: Array of optimal actions for each state
#         """
#         # TODO: For each state:
#         #       - Compute Q-values for all actions
#         #       - Select action with maximum Q-value
#         # TODO: Return policy array

#         raise NotImplementedError

#     def bellman_backup(self, state: int, values: np.ndarray) -> float:
#         """
#         Perform Bellman backup for a single state.

#         Args:
#             state: State index
#             values: Current value function

#         Returns:
#             Updated value for state
#         """
#         # TODO: If terminal state, return 0
#         # TODO: Compute Q-values for all actions
#         # TODO: Return maximum Q-value

#         raise NotImplementedError

#     def compute_bellman_error(self, values: np.ndarray) -> float:
#         """
#         Compute Bellman error for current value function.

#         Bellman error = max_s |V(s) - max_a Q(s,a)|

#         Args:
#             values: Current value function

#         Returns:
#             Maximum Bellman error across all states
#         """
#         # TODO: For each state:
#         #       - Compute optimal value using Bellman backup
#         #       - Calculate absolute difference from current value
#         # TODO: Return maximum error

#         raise NotImplementedError



"""
Value Iteration algorithm for solving MDPs.
"""

import numpy as np
from typing import Tuple, Optional
from environment import GridWorldEnv


class ValueIteration:
    """
    Value Iteration solver for gridworld MDP.

    Computes optimal value function V* using dynamic programming.
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
        """
        Initialize Value Iteration solver.

        Args:
            env: GridWorld environment
            gamma: Discount factor (default 0.95)
            epsilon: Convergence threshold (default 1e-4)
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = env.grid_size ** 2
        self.n_actions = env.action_space

    def solve(self, max_iterations: int = 1000) -> Tuple[np.ndarray, int]:
        """
        Run value iteration until convergence.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            values: Converged value function V(s)
            n_iterations: Number of iterations until convergence
        """
        # Initialize value function to zeros
        V = np.zeros(self.n_states, dtype=np.float32)
        n_iterations = 0

        for k in range(max_iterations):
            V_old = V.copy()
            
            # Perform a full sweep over all states
            for s in range(self.n_states):
                V[s] = self.bellman_backup(s, V_old)
            
            # Check convergence: max|V_new - V_old| < epsilon
            # The convergence criterion is specified as:
            # ||V_{k+1} - V_k||_\infty = max_s |V_{k+1}(s) - V_k(s)| < epsilon
            delta = np.max(np.abs(V - V_old))
            n_iterations = k + 1

            if delta < self.epsilon:
                break
        
        return V, n_iterations

    def compute_q_values(self, state: int, values: np.ndarray) -> np.ndarray:
        """
        Compute Q-values for all actions in a state.

        Args:
            state: State index
            values: Current value function V_k

        Returns:
            q_values: Array of Q(s,a) for each action
        """
        q_values = np.zeros(self.n_actions, dtype=np.float32)

        for a in range(self.n_actions):
            # Get transition probabilities P(s'|s,a)
            transitions = self.env.get_transition_prob(state, a)
            
            # Compute expected value: Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * V(s')]
            expected_return = 0.0
            for next_state, prob in transitions.items():
                reward = self.env.get_reward(state, a, next_state)
                expected_return += prob * (reward + self.gamma * values[next_state])
            
            q_values[a] = expected_return
            
        return q_values

    def extract_policy(self, values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from value function.
        
        Policy extraction: $\pi^*(s) = \arg\max_a Q^*(s, a)$

        Args:
            values: Optimal value function $V^*$

        Returns:
            policy: Array of optimal actions for each state
        """
        policy = np.zeros(self.n_states, dtype=int)
        
        for s in range(self.n_states):
            # If terminal state, we can arbitrarily assign an action (or a special terminal value)
            if self.env.is_terminal(s):
                policy[s] = -1 # Special value for terminal state
                continue

            # Compute Q-values for all actions based on $V^*$
            q_values = self.compute_q_values(s, values)
            
            # Select action with maximum Q-value
            policy[s] = np.argmax(q_values)
            
        return policy

    def bellman_backup(self, state: int, values: np.ndarray) -> float:
        """
        Perform Bellman backup for a single state: $V_{k+1}(s) = \max_a Q_k(s, a)$

        Args:
            state: State index
            values: Current value function $V_k$

        Returns:
            Updated value for state $V_{k+1}(s)$
        """
        # If terminal state (Goal), the value is fixed to 0, 
        # as per standard VI where the goal reward is received upon entry.
        # However, since the reward +10 is received upon *entering* G,
        # and the state is defined as terminal, we must check if V* should reflect
        # the immediate reward received OR if it's 0 (as in some conventions).
        # Given the VI formula, if the state is terminal, no future rewards matter. 
        # A common convention is V*(Goal) = 0.
        # Another approach is to compute Q-values and take the max, but since P(s'|s,a) for terminal s is P(s|s,a)=1, R(s,a,s)=0, this yields 0.
        
        if self.env.is_terminal(state):
            # A terminal state's value is often 0 (or defined by its reward, but since the
            # reward is from the *transition into* it, the state itself is 0 for future steps).
            return 0.0

        # Compute Q-values for all actions
        q_values = self.compute_q_values(state, values)
        
        # Return maximum Q-value
        return np.max(q_values)

    def compute_bellman_error(self, values: np.ndarray) -> float:
        """
        Compute Bellman error for current value function.

        Bellman error = max_s |V(s) - max_a Q(s,a)|

        Args:
            values: Current value function $V_k$

        Returns:
            Maximum Bellman error across all states
        """
        errors = []
        for s in range(self.n_states):
            optimal_value = self.bellman_backup(s, values)
            errors.append(np.abs(values[s] - optimal_value))
            
        return np.max(errors)