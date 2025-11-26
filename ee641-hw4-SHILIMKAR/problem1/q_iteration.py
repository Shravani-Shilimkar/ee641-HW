# """
# Q-Iteration algorithm for solving MDPs.
# """

# import numpy as np
# from typing import Tuple, Optional
# from environment import GridWorldEnv


# class QIteration:
#     """
#     Q-Iteration solver for gridworld MDP.

#     Computes optimal action-value function Q* using dynamic programming.
#     """

#     def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
#         """
#         Initialize Q-Iteration solver.

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
#         Run Q-iteration until convergence.

#         Args:
#             max_iterations: Maximum number of iterations

#         Returns:
#             q_values: Converged Q-function Q(s,a)
#             n_iterations: Number of iterations until convergence
#         """
#         # TODO: Initialize Q-function to zeros (shape: [n_states, n_actions])
#         # TODO: Iterate until convergence:
#         #       - For each state-action pair:
#         #           - Compute updated Q-value using Bellman equation:
#         #             Q(s,a) = sum_s' P(s'|s,a) * [R(s,a,s') + gamma * max_a' Q(s',a')]
#         #       - Check convergence: max|Q_new - Q_old| < epsilon
#         #       - Update Q-function
#         # TODO: Return final Q-values and iteration count

#         raise NotImplementedError

#     def bellman_update(self, state: int, action: int, q_values: np.ndarray) -> float:
#         """
#         Compute updated Q-value for a state-action pair.

#         Args:
#             state: State index
#             action: Action index
#             q_values: Current Q-function

#         Returns:
#             Updated Q-value for (s,a)
#         """
#         # TODO: Get transition probabilities P(s'|s,a)
#         # TODO: For each possible next state:
#         #       - Get reward R(s,a,s')
#         #       - Get max Q-value for next state: max_a' Q(s',a')
#         #       - Accumulate: prob * [reward + gamma * max_q_next]
#         # TODO: Return updated Q-value

#         raise NotImplementedError

#     def extract_policy(self, q_values: np.ndarray) -> np.ndarray:
#         """
#         Extract optimal policy from Q-function.

#         Args:
#             q_values: Optimal Q-function

#         Returns:
#             policy: Array of optimal actions for each state
#         """
#         # TODO: For each state:
#         #       - Select action with maximum Q-value: argmax_a Q(s,a)
#         # TODO: Return policy array

#         raise NotImplementedError

#     def extract_values(self, q_values: np.ndarray) -> np.ndarray:
#         """
#         Extract value function from Q-function.

#         Args:
#             q_values: Q-function

#         Returns:
#             values: State value function V(s) = max_a Q(s,a)
#         """
#         # TODO: For each state:
#         #       - Compute V(s) = max_a Q(s,a)
#         # TODO: Return value function

#         raise NotImplementedError

#     def compute_bellman_error(self, q_values: np.ndarray) -> float:
#         """
#         Compute Bellman error for current Q-function.

#         Args:
#             q_values: Current Q-function

#         Returns:
#             Maximum Bellman error across all state-action pairs
#         """
#         # TODO: For each state-action pair:
#         #       - Compute updated Q-value using Bellman update
#         #       - Calculate absolute difference from current Q-value
#         # TODO: Return maximum error

#         raise NotImplementedError



"""
Q-Iteration algorithm for solving MDPs.
"""

import numpy as np
from typing import Tuple, Optional
from environment import GridWorldEnv


class QIteration:
    """
    Q-Iteration solver for gridworld MDP.

    Computes optimal action-value function Q* using dynamic programming.
    """

    def __init__(self, env: GridWorldEnv, gamma: float = 0.95, epsilon: float = 1e-4):
        """
        Initialize Q-Iteration solver.

        Args:
            env: GridWorld environment
            gamma: Discount factor
            epsilon: Convergence threshold
        """
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_states = env.grid_size ** 2
        self.n_actions = env.action_space

    def solve(self, max_iterations: int = 1000) -> Tuple[np.ndarray, int]:
        """
        Run Q-iteration until convergence.

        Args:
            max_iterations: Maximum number of iterations

        Returns:
            q_values: Converged Q-function Q(s,a)
            n_iterations: Number of iterations until convergence
        """
        # Initialize Q-function to zeros (shape: [n_states, n_actions])
        Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        n_iterations = 0
        
        for k in range(max_iterations):
            Q_old = Q.copy()
            max_delta = 0.0

            # Iterate over all state-action pairs
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    # Compute updated Q-value using Bellman update
                    Q_new_sa = self.bellman_update(s, a, Q_old)
                    
                    # Calculate the difference for convergence check
                    delta = np.abs(Q_new_sa - Q_old[s, a])
                    if delta > max_delta:
                        max_delta = delta
                    
                    # Update Q-function
                    Q[s, a] = Q_new_sa

            # Check convergence: max|Q_new - Q_old| < epsilon
            # The convergence criterion is $\Vert Q_{k+1} - Q_k \Vert_\infty < \epsilon$
            n_iterations = k + 1
            if max_delta < self.epsilon:
                break
        
        return Q, n_iterations

    def bellman_update(self, state: int, action: int, q_values: np.ndarray) -> float:
        """
        Compute updated Q-value for a state-action pair: 
        $Q_{k+1}(s, a) = \sum_{s'} P(s'|s, a) [R(s, a, s') + \gamma \max_{a'} Q_k(s', a')]$

        Args:
            state: State index
            action: Action index
            q_values: Current Q-function $Q_k$

        Returns:
            Updated Q-value for $(s, a)$
        """
        # If terminal state (Goal), Q-values are 0 regardless of action, since no further steps are taken.
        if self.env.is_terminal(state):
            return 0.0

        expected_return = 0.0
        
        # Get transition probabilities $P(s'|s, a)$
        transitions = self.env.get_transition_prob(state, action)
        
        for next_state, prob in transitions.items():
            # 1. Get reward $R(s, a, s')$
            reward = self.env.get_reward(state, action, next_state)
            
            # 2. Get max Q-value for next state: $\max_{a'} Q_k(s', a')$
            # If next_state is terminal, its future value (max_a' Q(s', a')) is 0
            if self.env.is_terminal(next_state):
                max_q_next = 0.0
            else:
                # The maximum value over actions in the next state
                max_q_next = np.max(q_values[next_state, :])
            
            # 3. Accumulate: prob * [reward + gamma * max_q_next]
            expected_return += prob * (reward + self.gamma * max_q_next)
            
        return expected_return

    def extract_policy(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract optimal policy from Q-function: $\pi^*(s) = \arg\max_a Q^*(s, a)$

        Args:
            q_values: Optimal Q-function $Q^*$

        Returns:
            policy: Array of optimal actions for each state
        """
        # Select action with maximum Q-value for each state
        policy = np.argmax(q_values, axis=1)
        
        # Optionally, mark terminal state policy with a special value
        for s in range(self.n_states):
            if self.env.is_terminal(s):
                policy[s] = -1 # Special value for terminal state
                
        return policy

    def extract_values(self, q_values: np.ndarray) -> np.ndarray:
        """
        Extract value function from Q-function: $V(s) = \max_a Q(s, a)$

        Args:
            q_values: Q-function

        Returns:
            values: State value function $V(s)$
        """
        # Compute V(s) = max_a Q(s,a) for all states
        values = np.max(q_values, axis=1)
        
        # Terminal state value correction: V*(Goal) = 0
        for s in range(self.n_states):
             if self.env.is_terminal(s):
                values[s] = 0.0 
        
        return values

    def compute_bellman_error(self, q_values: np.ndarray) -> float:
        """
        Compute Bellman error for current Q-function.

        Bellman error = $\max_{s,a} |Q(s, a) - Q_{new}(s, a)|$

        Args:
            q_values: Current Q-function

        Returns:
            Maximum Bellman error across all state-action pairs
        """
        max_error = 0.0
        
        for s in range(self.n_states):
            # Skip terminal state since its value is fixed at 0 and doesn't propagate error
            if self.env.is_terminal(s):
                continue
                
            for a in range(self.n_actions):
                # Compute updated Q-value using Bellman update
                Q_new_sa = self.bellman_update(s, a, q_values)
                
                # Calculate absolute difference from current Q-value
                error = np.abs(q_values[s, a] - Q_new_sa)
                
                if error > max_error:
                    max_error = error
                    
        return max_error