# Standalone QMIX Testing Guide

This document explains how to use the QMIX algorithm extracted from the deep-marl-toolkit repository to test on custom environments.

## Overview

QMIX is a value decomposition multi-agent reinforcement learning algorithm suitable for cooperative multi-agent tasks. This repository provides a standalone test script `test_qmix_custom_env.py` that makes it easy to test the QMIX algorithm in custom discrete action space environments.

## Core Files

- **test_qmix_custom_env.py**: Main test script with complete QMIX training and evaluation flow
- **test_qmix_quick.py**: Quick test script for validating setup
- **test_qmix_simple.py**: Basic test script to verify component initialization
- **README_QMIX_STANDALONE.md**: Chinese documentation

## Environment Requirements

Your custom environment must meet the following requirements:

### Required Attributes

```python
class CustomEnvironment:
    num_agents: int        # Number of agents
    n_actions: int         # Number of discrete actions per agent
    obs_dim: int          # Observation dimension
    state_dim: int        # Global state dimension (optional, can be None)
    episode_limit: int    # Maximum steps per episode
```

### Required Methods

```python
def reset(self) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Reset the environment
    
    Returns:
        obs: Observation array, shape (num_agents, obs_dim)
        state: Global state array, shape (state_dim,)
        info: Information dictionary
    """
    pass

def step(self, actions: np.ndarray) -> Tuple:
    """
    Execute one step
    
    Args:
        actions: Action array, shape (num_agents,)
        
    Returns:
        next_obs: Next observation, shape (num_agents, obs_dim)
        next_state: Next global state, shape (state_dim,)
        reward: Scalar reward
        terminated: Whether episode is terminated (bool)
        truncated: Whether episode is truncated (bool)
        info: Information dictionary
    """
    pass

def get_available_actions(self) -> np.ndarray:
    """
    Get available actions for each agent (optional)
    
    Returns:
        available_actions: Action mask, shape (num_agents, n_actions)
                          1 means available, 0 means unavailable
    """
    # If not implemented, all actions will be assumed available
    pass
```

## Usage

### Method 1: Using Default Configuration

```python
from test_qmix_custom_env import test_qmix_with_custom_env

# Create your custom environment
custom_env = YourCustomEnvironment()

# Test QMIX with default configuration
trained_agent = test_qmix_with_custom_env(custom_env)
```

### Method 2: Custom Configuration

```python
import argparse
import torch
from test_qmix_custom_env import CustomEnvWrapper, create_qmix_agent, run_train_episode
from marltoolkit.data import ReplayBuffer

# 1. Wrap your environment
env = CustomEnvWrapper(your_custom_env)

# 2. Configure parameters
args = argparse.Namespace(
    # Environment parameters (from environment)
    num_agents=env.num_agents,
    n_actions=env.n_actions,
    obs_dim=env.obs_dim,
    state_dim=env.state_dim,
    episode_limit=env.episode_limit,
    obs_shape=env.get_actor_input_shape(),
    state_shape=env.state_shape,
    action_shape=env.action_shape,
    reward_shape=env.reward_shape,
    done_shape=env.done_shape,
    actor_input_dim=env.obs_dim,  # Required: actor network input dimension
    
    # Network architecture parameters
    fc_hidden_dim=64,          # Fully connected layer hidden dimension
    rnn_hidden_dim=64,         # RNN hidden layer dimension
    mixing_embed_dim=32,       # Mixer network embedding dimension
    hypernet_layers=2,         # Number of hypernetwork layers
    hypernet_embed_dim=64,     # Hypernetwork embedding dimension
    
    # Training parameters
    gamma=0.99,                # Discount factor
    learning_rate=0.0005,      # Learning rate
    min_learning_rate=0.00001, # Minimum learning rate
    egreedy_exploration=1.0,   # Initial epsilon
    min_exploration=0.01,      # Minimum epsilon
    target_update_tau=0.05,    # Target network soft update coefficient
    target_update_interval=100,# Target network update interval
    learner_update_freq=3,     # Learner update frequency
    double_q=True,             # Use Double Q-learning
    clip_grad_norm=10,         # Gradient clipping
    
    # Replay buffer parameters
    replay_buffer_size=5000,   # Replay buffer size
    batch_size=32,             # Batch size
    memory_warmup_size=32,     # Warmup size
    
    # Training schedule
    total_steps=50000,         # Total training steps
    train_log_interval=20,     # Training log interval
    test_log_interval=100,     # Test log interval
    num_eval_episodes=5,       # Number of evaluation episodes
    
    # Device
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

# 3. Create components
rpm = ReplayBuffer(
    max_size=args.replay_buffer_size,
    num_agents=args.num_agents,
    episode_limit=args.episode_limit,
    obs_shape=args.obs_shape,
    state_shape=args.state_shape,
    action_shape=args.action_shape,
    reward_shape=args.reward_shape,
    done_shape=args.done_shape,
    device=args.device,
)

agent = create_qmix_agent(env, args)

# 4. Training loop
# See complete example in test_qmix_custom_env.py
```

## Example: Simple Multi-Agent Environment

```python
import numpy as np

class SimpleMultiAgentEnv:
    """A simple example environment"""
    
    def __init__(self):
        self.num_agents = 3        # 3 agents
        self.n_actions = 5         # 5 actions per agent
        self.obs_dim = 10          # 10-dimensional observation
        self.state_dim = 15        # 15-dimensional global state
        self.episode_limit = 50    # Max 50 steps per episode
        self._step_count = 0
        
    def reset(self):
        self._step_count = 0
        obs = np.random.randn(self.num_agents, self.obs_dim).astype(np.float32)
        state = np.random.randn(self.state_dim).astype(np.float32)
        info = {}
        return obs, state, info
    
    def step(self, actions):
        self._step_count += 1
        next_obs = np.random.randn(self.num_agents, self.obs_dim).astype(np.float32)
        next_state = np.random.randn(self.state_dim).astype(np.float32)
        
        # Calculate reward (example: random reward)
        reward = np.random.randn() * 0.1
        
        # Check if episode is done
        terminated = self._step_count >= self.episode_limit
        truncated = False
        
        # Optional: add win information
        info = {'battle_won': terminated and (np.random.rand() > 0.5)}
        
        return next_obs, next_state, reward, terminated, truncated, info
    
    def get_available_actions(self):
        # Return mask where all actions are available
        return np.ones((self.num_agents, self.n_actions), dtype=np.int8)

# Use the environment
if __name__ == "__main__":
    from test_qmix_custom_env import test_qmix_with_custom_env
    
    env = SimpleMultiAgentEnv()
    agent = test_qmix_with_custom_env(env)
```

## Quick Testing

Run quick tests to validate your setup:

```bash
# Test basic components
python test_qmix_simple.py

# Quick training test (short duration)
python test_qmix_quick.py

# Complete training test
python test_qmix_custom_env.py
```

## QMIX Algorithm Components

Core components of the QMIX algorithm:

1. **RNN Actor Model**: Each agent's Q-network, uses GRU cell to handle partial observability
2. **Mixer Model**: Combines individual agent Q-values into global Q-value, uses Hypernetwork to ensure monotonicity
3. **Experience Replay**: Stores complete episodes for training
4. **Epsilon-greedy Exploration**: Gradually decreases exploration rate during training

## Code Structure

This implementation strictly follows the QMIX implementation pattern from the marltoolkit repository:

- Uses `marltoolkit.agents.qmix_agent.QMixAgent`
- Uses `marltoolkit.modules.actors.RNNActorModel` as actor network
- Uses `marltoolkit.modules.mixers.QMixerModel` as mixer network
- Uses `marltoolkit.data.ReplayBuffer` for experience replay
- Follows the training flow from `scripts/main_qmix.py`

## Important Notes

1. **Discrete Action Space**: QMIX only works with discrete action spaces
2. **Cooperative Tasks**: QMIX is designed for fully cooperative multi-agent tasks
3. **Shared Reward**: Environment should return a single shared team reward
4. **Partial Observability**: QMIX can handle partially observable environments (via RNN)
5. **Memory Requirements**: Replay buffer stores complete episodes, be mindful of memory usage

## Saving and Loading Models

```python
# Save model
agent.save_model(save_dir='./models', 
                 actor_model_name='actor.th',
                 mixer_model_name='mixer.th',
                 opt_name='optimizer.th')

# Load model
agent.load_model(save_dir='./models',
                 actor_model_name='actor.th',
                 mixer_model_name='mixer.th',
                 opt_name='optimizer.th')
```

## Troubleshooting

### Issue: `AttributeError: 'RNNActorModel' object has no attribute 'actor_input_dim'`

**Solution**: Make sure to set the `actor_input_dim` attribute in args:
```python
args.actor_input_dim = env.obs_dim
```

### Issue: Training is unstable or not converging

**Possible Solutions**:
- Reduce learning rate
- Increase target network update interval
- Adjust epsilon decay rate
- Increase replay buffer size
- Check if reward design is appropriate

### Issue: Out of memory

**Solutions**:
- Reduce replay buffer size
- Reduce batch size
- Shorten episode limit

## References

- Original QMIX paper: [QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- Repository code: `marltoolkit/agents/qmix_agent.py`
- Example script: `scripts/main_qmix.py`

## License

This code follows the license of the original repository.
