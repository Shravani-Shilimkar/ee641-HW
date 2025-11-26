# """
# Multi-agent gridworld environment with partial observations and communication.
# """

# import numpy as np
# from typing import Tuple, Optional, List


# class MultiAgentEnv:
#     """
#     Two-agent cooperative gridworld with partial observations.

#     Agents must coordinate to simultaneously reach a target cell.
#     Each agent observes a 3x3 local patch and exchanges communication signals.
#     """

#     def __init__(self, grid_size: Tuple[int, int] = (10, 10), obs_window: int = 3,
#                  max_steps: int = 50, seed: Optional[int] = None):
#         """
#         Initialize multi-agent environment.

#         Args:
#             grid_size: Tuple defining grid dimensions (default 10x10)
#             obs_window: Size of local observation window (must be odd, default 3)
#             max_steps: Maximum steps per episode
#             seed: Random seed for reproducibility
#         """
#         self.grid_size = grid_size
#         self.obs_window = obs_window
#         self.max_steps = max_steps

#         if seed is not None:
#             np.random.seed(seed)

#         # Initialize grid components
#         self._initialize_grid()

#         # Agent state
#         self.agent_positions = [None, None]
#         self.comm_signals = [0.0, 0.0]
#         self.step_count = 0

#     def _initialize_grid(self) -> None:
#         """
#         Create grid with obstacles and target.

#         Grid values:
#         - 0: Free cell
#         - 1: Obstacle
#         - 2: Target
#         """
#         # TODO: Create empty grid of size grid_size
#         # TODO: Randomly place up to 6 obstacles (avoiding corners)
#         # TODO: Randomly place exactly 1 target cell
#         # TODO: Store grid as self.grid

#         raise NotImplementedError

#     def reset(self) -> Tuple[np.ndarray, np.ndarray]:
#         """
#         Reset environment to initial state.

#         Returns:
#             obs_A: Observation for Agent A (11-dimensional vector)
#             obs_B: Observation for Agent B (11-dimensional vector)

#         Observation format:
#         - Elements 0-8: Flattened 3x3 grid patch (row-major order)
#         - Element 9: Communication signal from other agent
#         - Element 10: Normalized L2 distance between agents
#         """
#         # TODO: Reset step counter
#         # TODO: Randomly place both agents on free cells (not obstacles or target)
#         # TODO: Initialize communication signals to 0.0
#         # TODO: Generate observations for both agents
#         # TODO: Return (obs_A, obs_B)

#         raise NotImplementedError

#     def step(self, action_A: int, action_B: int, comm_A: float, comm_B: float) -> \
#             Tuple[Tuple[np.ndarray, np.ndarray], float, bool]:
#         """
#         Execute one environment step.

#         Args:
#             action_A: Agent A's movement action (0:Up, 1:Down, 2:Left, 3:Right, 4:Stay)
#             action_B: Agent B's movement action
#             comm_A: Communication signal from Agent A to B
#             comm_B: Communication signal from Agent B to A

#         Returns:
#             observations: Tuple of (obs_A, obs_B), each 11-dimensional
#             reward: +10 if both agents at target, +2 if one agent at target, -0.1 per step
#             done: True if both agents at target or max steps reached
#         """
#         # TODO: Update agent positions based on actions
#         #       - Check boundaries and obstacles
#         #       - Invalid moves result in no position change
#         # TODO: Store new communication signals for next observation
#         # TODO: Check reward condition (both agents at target)
#         # TODO: Update step count and check termination
#         # TODO: Generate new observations with updated comm signals
#         # TODO: Return ((obs_A, obs_B), reward, done)

#         raise NotImplementedError

#     def _get_observation(self, agent_idx: int) -> np.ndarray:
#         """
#         Extract local observation for an agent.

#         Args:
#             agent_idx: Agent index (0 for A, 1 for B)

#         Returns:
#             observation: 10-dimensional vector
#         """
#         # TODO: Get agent position
#         # TODO: Extract 3x3 patch centered on agent
#         #       - Cells outside grid should be -1
#         #       - Use grid values (0: free, 1: obstacle, 2: target)
#         # TODO: Flatten patch to 9 elements
#         # TODO: Append communication signal from other agent
#         # TODO: Return 10-dimensional observation

#         raise NotImplementedError

#     def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
#         """
#         Check if position is valid (in bounds and not obstacle).

#         Args:
#             pos: (row, col) position

#         Returns:
#             True if valid position
#         """
#         # TODO: Check if position is within grid bounds
#         # TODO: Check if position is not an obstacle (grid value != 1)

#         raise NotImplementedError

#     def _apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
#         """
#         Apply movement action to position.

#         Args:
#             pos: Current position (row, col)
#             action: Movement action (0-4)

#         Returns:
#             new_pos: Updated position (stays same if invalid)
#         """
#         # TODO: Map action to position delta
#         #       0: Up (-1, 0)
#         #       1: Down (+1, 0)
#         #       2: Left (0, -1)
#         #       3: Right (0, +1)
#         #       4: Stay (0, 0)
#         # TODO: Calculate new position
#         # TODO: Return new position if valid, else return original position

#         raise NotImplementedError

#     def _find_free_cells(self) -> List[Tuple[int, int]]:
#         """
#         Find all free cells in the grid.

#         Returns:
#             List of (row, col) positions that are free
#         """
#         # TODO: Iterate through grid
#         # TODO: Collect positions where grid value is 0 (free)
#         # TODO: Return list of free positions

#         raise NotImplementedError

#     def render(self) -> None:
#         """
#         Render current environment state.
#         """
#         # TODO: Create visual representation of grid
#         # TODO: Show agent positions (A, B)
#         # TODO: Show target (T)
#         # TODO: Show obstacles (X)
#         # TODO: Display current communication values

#         raise NotImplementedError



"""
Multi-agent gridworld environment with partial observations and communication.
"""

import numpy as np
from typing import Tuple, Optional, List


class MultiAgentEnv:
    """
    Two-agent cooperative gridworld with partial observations.

    Agents must coordinate to simultaneously reach a target cell.
    Each agent observes an 11-dimensional vector:
    - Elements 0-8: Flattened 3x3 grid patch (row-major order)
    - Element 9: Communication signal from other agent
    - Element 10: Normalized L2 distance between agents
    """

    def __init__(self, grid_size: Tuple[int, int] = (10, 10), obs_window: int = 3,
                 max_steps: int = 50, seed: Optional[int] = None):
        """
        Initialize multi-agent environment.

        Args:
            grid_size: Tuple defining grid dimensions (default 10x10)
            obs_window: Size of local observation window (must be odd, default 3)
            max_steps: Maximum steps per episode
            seed: Random seed for reproducibility
        """
        if obs_window % 2 == 0:
            raise ValueError("obs_window must be odd.")

        self.grid_size = grid_size
        self.H, self.W = grid_size
        self.obs_window = obs_window
        self.max_steps = max_steps
        self.half_window = obs_window // 2

        if seed is not None:
            np.random.seed(seed)
            self._rng = np.random.default_rng(seed)
        else:
            self._rng = np.random.default_rng()

        # Pre-calculate normalization factor for distance
        self._max_dist_norm = np.sqrt(self.H**2 + self.W**2)

        # Initialize grid components
        self._initialize_grid()

        # Agent state
        self.agent_positions = [None, None]  # [(row_A, col_A), (row_B, col_B)]
        self.comm_signals = [0.0, 0.0]  # [comm_A_out, comm_B_out] from previous step
        self.target_pos = self._find_target()
        self.step_count = 0

    def _initialize_grid(self) -> None:
        """
        Create grid with obstacles and target.
        Based on the provided image, we will use a fixed layout for reproducibility
        and consistency with the problem's scenario image.

        Grid values:
        - 0: Free cell
        - 1: Obstacle
        - 2: Target
        """
        # Create empty 10x10 grid
        self.grid = np.zeros(self.grid_size, dtype=np.int32)

        # Target (T) at (8, 8)
        self.grid[8, 8] = 2

        # Obstacles (X) based on image_e191e2.png:
        obstacles = [
            (2, 3), (2, 4),  # Row 2
            (4, 1),          # Row 4
            (5, 6),          # Row 5
            (7, 2),          # Row 7
            (8, 7)           # Row 8
        ]
        for r, c in obstacles:
            self.grid[r, c] = 1

    def _find_target(self) -> Tuple[int, int]:
        """Find the target position in the grid."""
        target_locs = np.argwhere(self.grid == 2)
        if len(target_locs) == 0:
            raise RuntimeError("Target not found in the initialized grid.")
        return tuple(target_locs[0])

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if position is valid (in bounds and not obstacle).
        """
        r, c = pos
        # Check bounds
        if not (0 <= r < self.H and 0 <= c < self.W):
            return False
        # Check if not an obstacle
        if self.grid[r, c] == 1:
            return False
        return True

    def _apply_action(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        Apply movement action to position.
        """
        r, c = pos
        # Deltas: 0:Up, 1:Down, 2:Left, 3:Right, 4:Stay
        deltas = {
            0: (-1, 0),
            1: (+1, 0),
            2: (0, -1),
            3: (0, +1),
            4: (0, 0)
        }

        dr, dc = deltas.get(action, (0, 0)) # Default to Stay if action is invalid
        new_pos = (r + dr, c + dc)

        # Check if the new position is valid
        if self._is_valid_position(new_pos):
            return new_pos
        else:
            # If invalid (boundary or obstacle), the agent stays put
            return pos

    def _find_free_cells(self) -> List[Tuple[int, int]]:
        """
        Find all free cells in the grid that are NOT the target and NOT an obstacle.
        """
        free_cells = []
        for r in range(self.H):
            for c in range(self.W):
                # Free cell (0) is safe for starting position
                if self.grid[r, c] == 0:
                    free_cells.append((r, c))
        return free_cells

    def _get_normalized_distance(self) -> float:
        """
        Calculate and return the normalized L2 distance between Agent A and B.
        """
        (rA, cA) = self.agent_positions[0]
        (rB, cB) = self.agent_positions[1]

        # Euclidean distance
        l2_dist = np.sqrt((rA - rB)**2 + (cA - cB)**2)

        # Normalized distance (bounded in [0, 1])
        # Note: self._max_dist_norm is sqrt(H^2 + W^2)
        normalized_dist = l2_dist / self._max_dist_norm
        return float(normalized_dist)

    def _get_observation(self, agent_idx: int) -> np.ndarray:
        """
        Extract local observation for an agent, plus communication and distance.
        The full observation is 11-dimensional.
        """
        # Agent position
        r, c = self.agent_positions[agent_idx]

        # 1. 3x3 Grid Patch (Elements 0-8)
        patch = np.full((self.obs_window, self.obs_window), -1, dtype=np.float32)

        for i in range(self.obs_window):
            for j in range(self.obs_window):
                # Global coordinates for cell (i, j) in the observation window
                r_global, c_global = r + i - self.half_window, c + j - self.half_window

                if 0 <= r_global < self.H and 0 <= c_global < self.W:
                    # Cell is on the grid
                    patch[i, j] = self.grid[r_global, c_global]
                else:
                    # Cell is off the grid (-1) - already set by np.full
                    pass

        # Flatten the patch (row-major)
        grid_patch_flat = patch.flatten()

        # 2. Communication Signal (Element 9)
        # This is the communication received from the *other* agent.
        partner_idx = 1 - agent_idx
        comm_in = self.comm_signals[partner_idx]

        # 3. Normalized Distance (Element 10)
        dist = self._get_normalized_distance()

        # Combine into the 11D observation vector
        observation = np.concatenate([
            grid_patch_flat,
            np.array([comm_in, dist], dtype=np.float32)
        ])

        return observation

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment to initial state.
        """
        self.step_count = 0

        # 1. Find all available starting positions
        free_cells = self._find_free_cells()
        if len(free_cells) < 2:
             raise RuntimeError("Not enough free cells to place two agents.")

        # 2. Randomly select two distinct free cells for agents A and B
        initial_indices = self._rng.choice(len(free_cells), size=2, replace=False)
        self.agent_positions[0] = free_cells[initial_indices[0]] # Pos A
        self.agent_positions[1] = free_cells[initial_indices[1]] # Pos B

        # 3. Initialize communication signals (from the *previous* step) to 0.0
        self.comm_signals = [0.0, 0.0]

        # 4. Generate observations
        obs_A = self._get_observation(agent_idx=0)
        obs_B = self._get_observation(agent_idx=1)

        return obs_A, obs_B

    def step(self, action_A: int, action_B: int, comm_A: float, comm_B: float) -> \
            Tuple[Tuple[np.ndarray, np.ndarray], float, bool]:
        """
        Execute one environment step.
        """
        self.step_count += 1

        # 1. Update agent positions based on actions
        new_pos_A = self._apply_action(self.agent_positions[0], action_A)
        new_pos_B = self._apply_action(self.agent_positions[1], action_B)

        # Agent collision check (if agents move to the same *free* cell) is not explicitly
        # mentioned as an invalid move, so we allow agents to occupy the same cell.
        self.agent_positions[0] = new_pos_A
        self.agent_positions[1] = new_pos_B

        # 2. Store new communication signals (for the *next* observation)
        # We ensure they are bounded in [0, 1] as required by the communication output description.
        self.comm_signals[0] = np.clip(comm_A, 0.0, 1.0)
        self.comm_signals[1] = np.clip(comm_B, 0.0, 1.0)

        # 3. Check reward and termination
        at_target_A = (self.agent_positions[0] == self.target_pos)
        at_target_B = (self.agent_positions[1] == self.target_pos)

        # Base penalty
        reward = -0.1

        # Partial reward
        if at_target_A ^ at_target_B: # XOR (one agent at target, the other is not)
            reward += 2.0

        # Full coordination reward
        done = False
        if at_target_A and at_target_B:
            reward += 10.0 # Total reward: -0.1 + 10.0 = 9.9
            done = True

        # Check max steps
        if self.step_count >= self.max_steps:
            done = True
            
        # 4. Generate new observations with updated comm signals
        obs_A = self._get_observation(agent_idx=0)
        obs_B = self._get_observation(agent_idx=1)

        return (obs_A, obs_B), reward, done

    def render(self) -> None:
        """
        Render current environment state.
        """
        grid_display = np.copy(self.grid).astype(str)
        
        # Replace codes with characters
        grid_display[grid_display == '0'] = '.' # Free
        grid_display[grid_display == '1'] = 'X' # Obstacle
        grid_display[grid_display == '2'] = 'T' # Target

        # Place agents (A and B) - prioritizing Agent A if they are in the same spot
        pos_A = self.agent_positions[0]
        pos_B = self.agent_positions[1]

        # Agent B is drawn first, then A (A overwrites B if same position)
        if pos_B is not None:
             grid_display[pos_B] = 'B'
        if pos_A is not None:
             grid_display[pos_A] = 'A'

        print(f"--- Step: {self.step_count}/{self.max_steps} ---")
        print("Grid:")
        print('\n'.join([' '.join(row) for row in grid_display]))
        print(f"Pos A: {pos_A}, Pos B: {pos_B}, Target: {self.target_pos}")
        print(f"Comm A->B: {self.comm_signals[0]:.4f}, Comm B->A: {self.comm_signals[1]:.4f}")
        
        at_target_A = (pos_A == self.target_pos)
        at_target_B = (pos_B == self.target_pos)
        print(f"A at Target: {at_target_A}, B at Target: {at_target_B}")
        if at_target_A and at_target_B:
            print("COORDINATION SUCCESS!")