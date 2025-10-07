"""
Example: Custom Multi-Agent Environment for QMIX Testing

This file demonstrates how to create a custom multi-agent environment
that is compatible with the QMIX test script.

The example implements a simple cooperative navigation task where
multiple agents need to reach target positions while avoiding obstacles.
"""

import numpy as np
from typing import Tuple, Dict


class CooperativeNavigationEnv:
    """
    Example custom environment: Cooperative Navigation
    
    Multiple agents need to cooperatively navigate to targets.
    - Observations: Agent position, velocities, relative target positions
    - Actions: 5 discrete actions (up, down, left, right, stay)
    - Reward: Negative distance to targets, collision penalty
    - Episode ends: When all agents reach targets or max steps
    """
    
    def __init__(
        self,
        num_agents: int = 3,
        grid_size: int = 10,
        episode_limit: int = 50,
        target_radius: float = 0.5,
    ):
        """
        Initialize the cooperative navigation environment.
        
        Args:
            num_agents: Number of agents in the environment
            grid_size: Size of the grid world
            episode_limit: Maximum steps per episode
            target_radius: Radius to consider target reached
        """
        # Environment parameters
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.episode_limit = episode_limit
        self.target_radius = target_radius
        
        # Action space: 0=up, 1=down, 2=left, 3=right, 4=stay
        self.n_actions = 5
        self.action_map = {
            0: np.array([0, 1]),   # up
            1: np.array([0, -1]),  # down
            2: np.array([-1, 0]),  # left
            3: np.array([1, 0]),   # right
            4: np.array([0, 0]),   # stay
        }
        
        # Observation: [agent_x, agent_y, target_x, target_y, distances_to_others]
        # Each agent observes: pos(2) + target_pos(2) + other_agents_distances(num_agents-1)
        self.obs_dim = 2 + 2 + (num_agents - 1)
        
        # Global state: all agent positions + all target positions
        self.state_dim = num_agents * 2 + num_agents * 2
        
        # Episode tracking
        self._step_count = 0
        self._agent_positions = None
        self._target_positions = None
        self._agents_reached_target = None
        
    def reset(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Reset the environment to initial state."""
        self._step_count = 0
        
        # Randomly initialize agent positions
        self._agent_positions = np.random.uniform(
            0, self.grid_size, size=(self.num_agents, 2)
        ).astype(np.float32)
        
        # Randomly initialize target positions
        self._target_positions = np.random.uniform(
            0, self.grid_size, size=(self.num_agents, 2)
        ).astype(np.float32)
        
        # Track which agents reached their targets
        self._agents_reached_target = np.zeros(self.num_agents, dtype=bool)
        
        # Get initial observations
        obs = self._get_observations()
        state = self._get_state()
        info = {}
        
        return obs, state, info
    
    def step(self, actions: np.ndarray) -> Tuple:
        """
        Execute one step in the environment.
        
        Args:
            actions: Array of actions for each agent, shape (num_agents,)
            
        Returns:
            next_obs: Next observations
            next_state: Next global state
            reward: Scalar reward
            terminated: Whether episode is done
            truncated: Whether episode is truncated
            info: Additional information
        """
        self._step_count += 1
        
        # Execute actions for each agent
        for i, action in enumerate(actions):
            if not self._agents_reached_target[i]:
                # Move agent based on action
                movement = self.action_map[int(action)]
                new_pos = self._agent_positions[i] + movement
                
                # Clip to grid boundaries
                new_pos = np.clip(new_pos, 0, self.grid_size)
                self._agent_positions[i] = new_pos
        
        # Check if agents reached their targets
        for i in range(self.num_agents):
            dist_to_target = np.linalg.norm(
                self._agent_positions[i] - self._target_positions[i]
            )
            if dist_to_target < self.target_radius:
                self._agents_reached_target[i] = True
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination conditions
        all_reached = np.all(self._agents_reached_target)
        time_limit = self._step_count >= self.episode_limit
        terminated = all_reached or time_limit
        truncated = False
        
        # Get next observations and state
        next_obs = self._get_observations()
        next_state = self._get_state()
        
        # Info dictionary
        info = {
            'battle_won': all_reached,  # Win if all agents reached targets
            'agents_reached': np.sum(self._agents_reached_target),
        }
        
        return next_obs, next_state, reward, terminated, truncated, info
    
    def _get_observations(self) -> np.ndarray:
        """Get observations for all agents."""
        observations = []
        
        for i in range(self.num_agents):
            # Own position
            own_pos = self._agent_positions[i]
            
            # Target position (relative)
            target_pos = self._target_positions[i] - own_pos
            
            # Distances to other agents
            distances = []
            for j in range(self.num_agents):
                if i != j:
                    dist = np.linalg.norm(
                        self._agent_positions[i] - self._agent_positions[j]
                    )
                    distances.append(dist)
            
            # Combine into observation
            obs = np.concatenate([
                own_pos,
                target_pos,
                np.array(distances, dtype=np.float32)
            ])
            observations.append(obs)
        
        return np.array(observations, dtype=np.float32)
    
    def _get_state(self) -> np.ndarray:
        """Get global state."""
        # Global state includes all agent positions and all target positions
        state = np.concatenate([
            self._agent_positions.flatten(),
            self._target_positions.flatten()
        ])
        return state.astype(np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate team reward."""
        reward = 0.0
        
        # Reward based on distance to targets
        for i in range(self.num_agents):
            if not self._agents_reached_target[i]:
                dist = np.linalg.norm(
                    self._agent_positions[i] - self._target_positions[i]
                )
                # Negative distance as reward (closer is better)
                reward -= dist / self.grid_size
        
        # Bonus for reaching target
        newly_reached = np.sum(self._agents_reached_target)
        reward += newly_reached * 2.0
        
        # Small penalty for each step to encourage efficiency
        reward -= 0.01
        
        return reward
    
    def get_available_actions(self) -> np.ndarray:
        """Get available actions for each agent (all actions always available)."""
        return np.ones((self.num_agents, self.n_actions), dtype=np.int8)


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Testing Custom Cooperative Navigation Environment with QMIX")
    print("="*60)
    
    # Create the environment
    env = CooperativeNavigationEnv(
        num_agents=3,
        grid_size=10,
        episode_limit=50,
    )
    
    print(f"\nEnvironment Configuration:")
    print(f"  Number of agents: {env.num_agents}")
    print(f"  Number of actions: {env.n_actions}")
    print(f"  Observation dimension: {env.obs_dim}")
    print(f"  State dimension: {env.state_dim}")
    print(f"  Episode limit: {env.episode_limit}")
    
    # Test the environment manually
    print("\n" + "-"*60)
    print("Manual Environment Test")
    print("-"*60)
    
    obs, state, info = env.reset()
    print(f"\nInitial observations shape: {obs.shape}")
    print(f"Initial state shape: {state.shape}")
    
    # Run a few random steps
    for step in range(5):
        actions = np.random.randint(0, env.n_actions, size=env.num_agents)
        next_obs, next_state, reward, terminated, truncated, info = env.step(actions)
        print(f"\nStep {step+1}:")
        print(f"  Actions: {actions}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Agents reached target: {info['agents_reached']}")
        
        if terminated or truncated:
            print(f"  Episode ended!")
            break
    
    # Instructions for QMIX training
    print("\n" + "="*60)
    print("To train QMIX on this environment:")
    print("="*60)
    print("\nOption 1: Use the wrapper function")
    print("  from test_qmix_custom_env import test_qmix_with_custom_env")
    print("  env = CooperativeNavigationEnv()")
    print("  agent = test_qmix_with_custom_env(env)")
    print("\nOption 2: Custom training loop")
    print("  See test_qmix_custom_env.py for detailed example")
    print("\nOption 3: Quick test")
    print("  Modify test_qmix_quick.py to use CooperativeNavigationEnv")
    print("\nFor detailed instructions, see:")
    print("  - QMIX_USAGE.md (English documentation)")
    print("  - README_QMIX_STANDALONE.md (Chinese documentation)")
    print("\n" + "="*60)
