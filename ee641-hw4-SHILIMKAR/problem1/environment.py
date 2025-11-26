# """
# Stochastic gridworld environment for reinforcement learning.
# """

# import numpy as np
# from typing import Tuple, List, Optional, Dict


# class GridWorldEnv:
#     """
#     5x5 Stochastic GridWorld Environment.

#     The agent navigates a grid with stochastic transitions:
#     - 0.8 probability of moving in the intended direction
#     - 0.1 probability of drifting left (perpendicular)
#     - 0.1 probability of drifting right (perpendicular)

#     Grid layout:
#     - Start: (0, 0)
#     - Goal: (4, 4)
#     - Obstacles: (2, 2), (1, 3)
#     - Penalties: (3, 1), (0, 3)
#     """

#     def __init__(self, seed: Optional[int] = None):
#         """
#         Initialize gridworld environment.

#         Args:
#             seed: Random seed for reproducibility
#         """
#         self.grid_size = 5
#         self.max_steps = 50

#         # Define special cells
#         self.start_pos = (0, 0)
#         self.goal_pos = (4, 4)
#         self.obstacles = [(1, 2), (2, 1)]
#         self.penalties = [(3, 3), (3, 0)]

#         # Rewards
#         self.goal_reward = 10.0
#         self.penalty_reward = -5.0
#         self.step_cost = -0.1

#         # Transition probabilities
#         self.prob_intended = 0.8
#         self.prob_drift = 0.1

#         # Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
#         self.action_space = 4
#         self.action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']

#         if seed is not None:
#             np.random.seed(seed)

#         self.reset()

#     def reset(self) -> int:
#         """
#         Reset environment to initial state.

#         Returns:
#             state: Initial state index
#         """
#         # TODO: Initialize agent position to start_pos
#         # TODO: Reset step counter
#         # TODO: Set done flag to False
#         # TODO: Return state index (use _pos_to_state)

#         raise NotImplementedError

#     def step(self, action: int) -> Tuple[int, float, bool, Dict]:
#         """
#         Execute action in environment.

#         Args:
#             action: Action index (0-3)

#         Returns:
#             next_state: Next state index
#             reward: Reward received
#             done: Whether episode terminated
#             info: Additional information
#         """
#         # TODO: Check if episode already done
#         # TODO: Get next position based on stochastic transitions
#         # TODO: Calculate reward (use _calculate_reward helper)
#         # TODO: Update position and step count
#         # TODO: Check termination conditions
#         # TODO: Return (next_state, reward, done, info)

#         raise NotImplementedError

#     def get_transition_prob(self, state: int, action: int) -> Dict[int, float]:
#         """
#         Get transition probabilities P(s'|s,a).

#         Args:
#             state: Current state index
#             action: Action index

#         Returns:
#             Dictionary mapping next_state -> probability
#         """
#         # TODO: Convert state to position
#         # TODO: For given action, compute all possible next positions
#         #       considering stochastic transitions
#         # TODO: Handle boundary and obstacle collisions
#         # TODO: Return probability distribution over next states

#         raise NotImplementedError

#     def get_reward(self, state: int, action: int, next_state: int) -> float:
#         """
#         Get reward for transition.

#         Args:
#             state: Current state index
#             action: Action taken
#             next_state: Resulting state

#         Returns:
#             Reward value
#         """
#         # TODO: Convert next_state to position
#         # TODO: Check if goal reached (+10)
#         # TODO: Check if penalty cell (-5)
#         # TODO: Otherwise return step cost (-0.1)

#         raise NotImplementedError

#     def is_terminal(self, state: int) -> bool:
#         """
#         Check if state is terminal.

#         Args:
#             state: State index

#         Returns:
#             True if terminal state
#         """
#         # TODO: Convert state to position
#         # TODO: Return True if position equals goal_pos

#         raise NotImplementedError

#     def _pos_to_state(self, pos: Tuple[int, int]) -> int:
#         """
#         Convert grid position to state index.

#         Args:
#             pos: (row, col) position

#         Returns:
#             State index (0-24)
#         """
#         # TODO: Convert 2D position to 1D state index
#         # State = row * grid_size + col

#         raise NotImplementedError

#     def _state_to_pos(self, state: int) -> Tuple[int, int]:
#         """
#         Convert state index to grid position.

#         Args:
#             state: State index

#         Returns:
#             (row, col) position
#         """
#         # TODO: Convert 1D state index to 2D position
#         # row = state // grid_size
#         # col = state % grid_size

#         raise NotImplementedError

#     def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
#         """
#         Check if position is valid (in bounds and not obstacle).

#         Args:
#             pos: (row, col) position

#         Returns:
#             True if valid position
#         """
#         # TODO: Check if position is within grid bounds
#         # TODO: Check if position is not an obstacle

#         raise NotImplementedError

#     def _get_next_positions(self, pos: Tuple[int, int], action: int) -> List[Tuple[Tuple[int, int], float]]:
#         """
#         Get possible next positions and probabilities for stochastic transition.

#         Args:
#             pos: Current position
#             action: Action to take

#         Returns:
#             List of (next_position, probability) tuples
#         """
#         # TODO: Define action effects (deltas for UP, RIGHT, DOWN, LEFT)
#         # TODO: Get intended direction and perpendicular directions
#         # TODO: For each possible outcome (intended, drift left, drift right):
#         #       - Calculate next position
#         #       - If invalid, stay in current position
#         #       - Add (position, probability) to list
#         # TODO: Merge probabilities for same positions

#         raise NotImplementedError

#     def _calculate_reward(self, pos: Tuple[int, int]) -> float:
#         """
#         Calculate reward for entering a position.

#         Args:
#             pos: Position entered

#         Returns:
#             Reward value
#         """
#         # TODO: Check if position is goal (+10)
#         # TODO: Check if position is penalty (-5)
#         # TODO: Otherwise return step cost (-0.1)

#         raise NotImplementedError

#     def render(self, value_function: Optional[np.ndarray] = None) -> None:
#         """
#         Render current state of environment.

#         Args:
#             value_function: Optional value function to display
#         """
#         # TODO: Create visual representation of grid
#         # TODO: Mark current position, goal, obstacles, penalties
#         # TODO: If value_function provided, show as heatmap

#         raise NotImplementedError



"""
Stochastic gridworld environment for reinforcement learning.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Union


class GridWorldEnv:
    """
    5x5 Stochastic GridWorld Environment.

    The agent navigates a grid with stochastic transitions:
    - 0.8 probability of moving in the intended direction
    - 0.1 probability of drifting left (perpendicular)
    - 0.1 probability of drifting right (perpendicular)

    Grid layout:
    - Start: (0, 0)
    - Goal: (4, 4)
    - Obstacles: (1, 2), (2, 1)  # Corrected from original comment to match image
    - Penalties: (3, 0), (3, 3)  # Corrected from original comment to match image
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize gridworld environment.

        Args:
            seed: Random seed for reproducibility
        """
        self.grid_size = 5
        self.max_steps = 50
        self.state_space = self.grid_size * self.grid_size

        # Define special cells (Row, Col) based on the Grid Environment image
        self.start_pos = (0, 0)
        self.goal_pos = (4, 4)
        # X: (1, 2), (2, 1)
        self.obstacles = [(1, 2), (2, 1)] 
        # P: (3, 0), (3, 3)
        self.penalties = [(3, 0), (3, 3)] 

        # Rewards
        self.goal_reward = 10.0
        self.penalty_reward = -5.0
        self.step_cost = -0.1

        # Transition probabilities
        self.prob_intended = 0.8
        self.prob_drift = 0.1

        # Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = 4
        self.action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        
        # Delta row/col for each action (0=UP, 1=RIGHT, 2=DOWN, 3=LEFT)
        self.action_deltas = [
            (-1, 0),  # UP
            (0, 1),   # RIGHT
            (1, 0),   # DOWN
            (0, -1)   # LEFT
        ]

        if seed is not None:
            np.random.seed(seed)
            self.np_random = np.random.RandomState(seed)
        else:
            self.np_random = np.random.RandomState()

        self.reset()

    def reset(self) -> int:
        """
        Reset environment to initial state.

        Returns:
            state: Initial state index
        """
        self.current_pos = self.start_pos
        self.steps = 0
        self.done = False
        return self._pos_to_state(self.current_pos)

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: Action index (0-3)

        Returns:
            next_state: Next state index
            reward: Reward received
            done: Whether episode terminated
            info: Additional information
        """
        if self.done:
            return self._pos_to_state(self.current_pos), 0.0, True, {"steps": self.steps}

        # 1. Get possible next positions and probabilities
        possible_transitions = self._get_next_positions(self.current_pos, action)
        
        # 2. Select next position based on probabilities (for actual step execution)
        positions, probs = zip(*possible_transitions)
        
        # The choice must be a tuple of (row, col)
        if len(positions) > 0:
            next_pos_idx = self.np_random.choice(len(positions), p=probs)
            next_pos = positions[next_pos_idx]
        else:
            # Should not happen if _get_next_positions is correct, but safety
            next_pos = self.current_pos 

        # 3. Calculate reward
        reward = self._calculate_reward(next_pos)

        # 4. Update position and step count
        self.current_pos = next_pos
        self.steps += 1

        # 5. Check termination conditions
        if self.is_terminal(self._pos_to_state(next_pos)) or self.steps >= self.max_steps:
            self.done = True

        next_state = self._pos_to_state(next_pos)
        info = {"steps": self.steps}

        return next_state, reward, self.done, info

    def get_transition_prob(self, state: int, action: int) -> Dict[int, float]:
        """
        Get transition probabilities P(s'|s,a). Used for Value/Q-Iteration.

        Args:
            state: Current state index
            action: Action index

        Returns:
            Dictionary mapping next_state -> probability
        """
        pos = self._state_to_pos(state)
        
        # If already at a terminal state, the only next state is itself with probability 1.
        if self.is_terminal(state):
            return {state: 1.0}

        # Get next positions and probabilities (merging duplicates)
        pos_probs = self._get_next_positions(pos, action)
        
        transition_probs = {}
        for next_pos, prob in pos_probs:
            next_state = self._pos_to_state(next_pos)
            transition_probs[next_state] = transition_probs.get(next_state, 0.0) + prob

        return transition_probs

    def get_reward(self, state: int, action: int, next_state: int) -> float:
        """
        Get reward for transition. Used for Value/Q-Iteration.
        Note: The problem description implies the reward R(s, a, s') is simply R(s').

        Args:
            state: Current state index (not used)
            action: Action taken (not used)
            next_state: Resulting state

        Returns:
            Reward value
        """
        next_pos = self._state_to_pos(next_state)
        return self._calculate_reward(next_pos)

    def is_terminal(self, state: int) -> bool:
        """
        Check if state is terminal.

        Args:
            state: State index

        Returns:
            True if terminal state
        """
        pos = self._state_to_pos(state)
        return pos == self.goal_pos

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        """
        Convert grid position to state index.

        Args:
            pos: (row, col) position

        Returns:
            State index (0-24)
        """
        row, col = pos
        return row * self.grid_size + col

    def _state_to_pos(self, state: int) -> Tuple[int, int]:
        """
        Convert state index to grid position.

        Args:
            state: State index

        Returns:
            (row, col) position
        """
        row = state // self.grid_size
        col = state % self.grid_size
        return (row, col)

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).

        Args:
            pos: (row, col) position

        Returns:
            True if valid position
        """
        row, col = pos
        # Check bounds
        if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
            return False
        # Check obstacle
        if pos in self.obstacles:
            return False
        return True

    def _get_next_positions(self, pos: Tuple[int, int], action: int) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get possible next positions and probabilities for stochastic transition.

        Args:
            pos: Current position
            action: Action to take

        Returns:
            List of (next_position, probability) tuples
        """
        
        # 1. Determine intended, left-drift, and right-drift actions
        intended_action = action
        # Left-drift is one index counter-clockwise (e.g., UP(0) -> LEFT(3))
        drift_left_action = (action - 1) % self.action_space 
        # Right-drift is one index clockwise (e.g., UP(0) -> RIGHT(1))
        drift_right_action = (action + 1) % self.action_space 
        
        # Map outcomes to their probabilities
        outcomes = [
            (intended_action, self.prob_intended),
            (drift_left_action, self.prob_drift),
            (drift_right_action, self.prob_drift)
        ]
        
        row, col = pos
        raw_transitions = []
        
        for move_action, prob in outcomes:
            dr, dc = self.action_deltas[move_action]
            next_row, next_col = row + dr, col + dc
            next_pos_candidate = (next_row, next_col)
            
            # 2. Collision handling: If invalid, agent stays at current position
            if self._is_valid_pos(next_pos_candidate):
                final_next_pos = next_pos_candidate
            else:
                final_next_pos = pos  # Stays in current state
            
            raw_transitions.append((final_next_pos, prob))

        # 3. Merge probabilities for the same next position
        merged_transitions: Dict[Tuple[int, int], float] = {}
        for next_pos, prob in raw_transitions:
            merged_transitions[next_pos] = merged_transitions.get(next_pos, 0.0) + prob
            
        # Convert back to the required list format
        return list(merged_transitions.items())

    def _calculate_reward(self, pos: Tuple[int, int]) -> float:
        """
        Calculate reward for entering a position.

        Args:
            pos: Position entered

        Returns:
            Reward value
        """
        if pos == self.goal_pos:
            return self.goal_reward # +10
        elif pos in self.penalties:
            return self.penalty_reward # -5
        else:
            return self.step_cost # -0.1

    def render(self, value_function: Optional[np.ndarray] = None) -> None:
        """
        Render current state of environment. (Basic text rendering for simplicity)

        Args:
            value_function: Optional value function to display (not fully implemented here)
        """
        grid = np.full((self.grid_size, self.grid_size), '.')
        
        # Mark special cells
        grid[self.goal_pos] = 'G'
        for r, c in self.obstacles: grid[r, c] = 'X'
        for r, c in self.penalties: grid[r, c] = 'P'
        
        # Mark current position
        if not self.done:
            grid[self.current_pos] = 'A' # Agent
        
        print("\n--- GridWorld State ---")
        for r in range(self.grid_size):
            row_str = " ".join(grid[r])
            if value_function is not None:
                # Add value function next to the cell
                values = [f"{value_function[self._pos_to_state((r, c))]:.2f}" for c in range(self.grid_size)]
                row_str += " | " + " ".join(values)
            print(row_str)
        print(f"Current Pos: {self.current_pos}, Steps: {self.steps}, Done: {self.done}")
        if value_function is not None:
            print("--- GridWorld State (Value Function) ---")
        else:
            print("-----------------------")