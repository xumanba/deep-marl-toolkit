"""
Standalone QMIX Test Script for Custom Environments
This script extracts QMIX algorithm from the marltoolkit repository
and provides a simple interface for testing on custom discrete action space environments.

Based on the implementation in marltoolkit repository.
"""

import argparse
import numpy as np
import torch
from typing import Tuple

# Import from the marltoolkit repository
from marltoolkit.agents.qmix_agent import QMixAgent
from marltoolkit.data import ReplayBuffer
from marltoolkit.modules.actors import RNNActorModel
from marltoolkit.modules.mixers import QMixerModel


class CustomEnvWrapper:
    """
    A simple wrapper for custom multi-agent environments with discrete action space.
    
    Your custom environment should provide:
    - num_agents: number of agents
    - n_actions: number of discrete actions per agent
    - obs_dim: observation dimension for each agent
    - state_dim: global state dimension (optional, can be None)
    - episode_limit: maximum steps per episode
    
    Methods required:
    - reset() -> (obs, state, info)
    - step(actions) -> (next_obs, next_state, reward, terminated, truncated, info)
    - get_available_actions() -> available_actions array
    """
    
    def __init__(self, env):
        """
        Initialize wrapper for custom environment.
        
        Args:
            env: Your custom environment instance
        """
        self.env = env
        
        # Get environment information
        self.num_agents = env.num_agents
        self.n_actions = env.n_actions
        self.obs_dim = env.obs_dim
        self.state_dim = env.state_dim if hasattr(env, 'state_dim') else None
        self.episode_limit = env.episode_limit
        
        # Define shapes
        self.obs_shape = (self.obs_dim,)
        self.action_shape = (self.n_actions,)
        self.state_shape = (self.state_dim,) if self.state_dim is not None else None
        self.reward_shape = (1,)
        self.done_shape = (1,)
        
        # Win counting (optional, for evaluation)
        self.win_counted = False
        
    def reset(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Reset the environment."""
        obs, state, info = self.env.reset()
        self.win_counted = False
        return obs, state, info
    
    def step(self, actions: np.ndarray):
        """Execute one step in the environment."""
        next_obs, next_state, reward, terminated, truncated, info = self.env.step(actions)
        
        # Update win status if available
        if 'battle_won' in info:
            self.win_counted = info['battle_won']
        elif 'is_win' in info:
            self.win_counted = info['is_win']
            
        return next_obs, next_state, reward, terminated, truncated, info
    
    def get_available_actions(self) -> np.ndarray:
        """Get available actions for each agent."""
        if hasattr(self.env, 'get_available_actions'):
            return self.env.get_available_actions()
        else:
            # If not provided, assume all actions are available
            return np.ones((self.num_agents, self.n_actions), dtype=np.int8)
    
    def get_actor_input_shape(self) -> Tuple:
        """Get the input shape for actor network."""
        input_dim = self.obs_dim
        return (input_dim,)


def run_train_episode(
    env: CustomEnvWrapper,
    agent: QMixAgent,
    rpm: ReplayBuffer,
    args: argparse.Namespace,
) -> dict:
    """
    Run one training episode (following marltoolkit repository pattern).
    
    Args:
        env: Environment wrapper
        agent: QMIX agent
        rpm: Replay buffer
        args: Configuration arguments
        
    Returns:
        Dictionary containing episode statistics
    """
    episode_reward = 0.0
    episode_step = 0
    done = False
    
    # Initialize hidden states for RNN
    agent.init_hidden_states(batch_size=1)
    
    # Reset environment
    obs, state, info = env.reset()
    
    # Episode loop
    while not done:
        # Get available actions
        available_actions = env.get_available_actions()
        
        # Sample actions from agent
        actions = agent.sample(obs=obs, available_actions=available_actions)
        
        # Execute actions in environment
        next_obs, next_state, reward, terminated, truncated, info = env.step(actions)
        done = terminated or truncated
        
        # Store transition
        transitions = {
            'obs': obs,
            'state': state,
            'actions': actions,
            'available_actions': available_actions,
            'rewards': reward,
            'dones': done,
            'filled': False,
        }
        rpm.store_transitions(transitions)
        
        # Update for next step
        obs, state = next_obs, next_state
        episode_reward += reward
        episode_step += 1
    
    # Fill remaining episode steps (padding for fixed-length episodes)
    for _ in range(episode_step, args.episode_limit):
        rpm.episode_data.fill_mask()
    
    # Store complete episode in replay buffer
    rpm.store_episodes()
    is_win = env.win_counted
    
    # Training phase
    train_res_lst = []
    if rpm.size() > args.memory_warmup_size:
        for _ in range(args.learner_update_freq):
            batch = rpm.sample(args.batch_size)
            results = agent.learn(batch)
            train_res_lst.append(results)
    
    # Aggregate training results
    if train_res_lst:
        train_res_dict = {
            'loss': np.mean([r['loss'] for r in train_res_lst]),
            'mean_td_error': np.mean([r['mean_td_error'] for r in train_res_lst])
        }
    else:
        train_res_dict = {'loss': None, 'mean_td_error': None}
    
    train_res_dict['episode_reward'] = episode_reward
    train_res_dict['episode_step'] = episode_step
    train_res_dict['win_rate'] = is_win
    
    return train_res_dict


def run_eval_episode(
    env: CustomEnvWrapper,
    agent: QMixAgent,
    args: argparse.Namespace,
) -> dict:
    """
    Run evaluation episodes (following marltoolkit repository pattern).
    
    Args:
        env: Environment wrapper
        agent: QMIX agent
        args: Configuration arguments
        
    Returns:
        Dictionary containing evaluation statistics
    """
    eval_res_list = []
    
    for _ in range(args.num_eval_episodes):
        agent.init_hidden_states(batch_size=1)
        episode_reward = 0.0
        episode_step = 0
        done = False
        obs, state, info = env.reset()
        
        while not done:
            available_actions = env.get_available_actions()
            # Use greedy policy for evaluation
            actions = agent.predict(obs=obs, available_actions=available_actions)
            next_obs, next_state, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            obs = next_obs
            episode_step += 1
            episode_reward += reward
        
        is_win = env.win_counted
        eval_res_list.append({
            'episode_reward': episode_reward,
            'episode_step': episode_step,
            'win_rate': is_win,
        })
    
    # Average over evaluation episodes
    eval_res_dict = {
        'episode_reward': np.mean([r['episode_reward'] for r in eval_res_list]),
        'episode_step': np.mean([r['episode_step'] for r in eval_res_list]),
        'win_rate': np.mean([r['win_rate'] for r in eval_res_list]),
    }
    
    return eval_res_dict


def create_qmix_agent(env_wrapper, args):
    """
    Create QMIX agent with actor and mixer models.
    Following the pattern from marltoolkit/scripts/main_qmix.py
    
    Args:
        env_wrapper: Environment wrapper
        args: Configuration arguments
        
    Returns:
        Initialized QMIX agent
    """
    # Set actor_input_dim for RNNActorModel (required by the model)
    args.actor_input_dim = args.obs_shape[0]
    
    # Create actor model (RNN-based Q-network for each agent)
    actor_model = RNNActorModel(args)
    
    # Create mixer model (combines individual Q-values into global Q)
    mixer_model = QMixerModel(
        num_agents=args.num_agents,
        state_dim=args.state_dim,
        mixing_embed_dim=args.mixing_embed_dim,
        hypernet_layers=args.hypernet_layers,
        hypernet_embed_dim=args.hypernet_embed_dim,
    )
    
    # Create QMIX agent
    agent = QMixAgent(
        actor_model=actor_model,
        mixer_model=mixer_model,
        num_agents=args.num_agents,
        double_q=args.double_q,
        total_steps=args.total_steps,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        min_learning_rate=args.min_learning_rate,
        egreedy_exploration=args.egreedy_exploration,
        min_exploration=args.min_exploration,
        target_update_tau=args.target_update_tau,
        target_update_interval=args.target_update_interval,
        learner_update_freq=args.learner_update_freq,
        clip_grad_norm=args.clip_grad_norm,
        device=args.device,
    )
    
    return agent


def test_qmix_with_custom_env(custom_env):
    """
    Main function to test QMIX on a custom environment.
    This follows the structure from marltoolkit/scripts/main_qmix.py
    
    Args:
        custom_env: Your custom multi-agent environment instance
    """
    # Wrap the custom environment
    env = CustomEnvWrapper(custom_env)
    
    # Configuration (based on configs/qmix_config.py)
    args = argparse.Namespace(
        # Environment parameters (from env)
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
        
        # Network architecture parameters
        fc_hidden_dim=64,
        rnn_hidden_dim=64,
        mixing_embed_dim=32,
        hypernet_layers=2,
        hypernet_embed_dim=64,
        
        # Training parameters
        gamma=0.99,
        learning_rate=0.0005,
        min_learning_rate=0.00001,
        egreedy_exploration=1.0,
        min_exploration=0.01,
        target_update_tau=0.05,
        target_update_interval=100,
        learner_update_freq=3,
        double_q=True,
        clip_grad_norm=10,
        
        # Replay buffer parameters
        replay_buffer_size=5000,
        batch_size=32,
        memory_warmup_size=32,
        
        # Training schedule
        total_steps=50000,
        train_log_interval=20,
        test_log_interval=100,
        num_eval_episodes=5,
        
        # Device
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    
    print("=" * 60)
    print("QMIX Training on Custom Environment")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Number of agents: {args.num_agents}")
    print(f"Number of actions: {args.n_actions}")
    print(f"Observation dim: {args.obs_dim}")
    print(f"State dim: {args.state_dim}")
    print(f"Episode limit: {args.episode_limit}")
    print("=" * 60)
    
    # Create replay buffer
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
    
    # Create QMIX agent
    agent = create_qmix_agent(env, args)
    
    # Warmup phase: fill replay buffer
    print("\n[Warmup Phase] Filling replay buffer...")
    warmup_cnt = 0
    while rpm.size() < args.memory_warmup_size:
        run_train_episode(env, agent, rpm, args)
        warmup_cnt += 1
        if warmup_cnt % 10 == 0:
            print(f"  Warmup episodes: {warmup_cnt}, Buffer size: {rpm.size()}")
    
    print(f"[Warmup Phase] Completed with {rpm.size()} transitions\n")
    
    # Training loop
    print("[Training Phase] Starting training...")
    steps_cnt = 0
    episode_cnt = 0
    
    while steps_cnt < args.total_steps:
        # Run training episode
        train_res_dict = run_train_episode(env, agent, rpm, args)
        
        # Update counters
        episode_cnt += 1
        steps_cnt += train_res_dict['episode_step']
        
        # Update learning rate (decay)
        agent.learning_rate = max(
            agent.lr_scheduler.step(train_res_dict['episode_step']),
            agent.min_learning_rate,
        )
        
        # Logging
        if episode_cnt % args.train_log_interval == 0:
            loss_str = f"{train_res_dict['loss']:.4f}" if train_res_dict['loss'] is not None else "N/A"
            td_error_str = f"{train_res_dict['mean_td_error']:.4f}" if train_res_dict['mean_td_error'] is not None else "N/A"
            
            print(f"[Train] Episode: {episode_cnt:4d}, "
                  f"Steps: {steps_cnt:6d}, "
                  f"Reward: {train_res_dict['episode_reward']:.2f}, "
                  f"Win: {train_res_dict['win_rate']:.2f}, "
                  f"Loss: {loss_str}, "
                  f"Exploration: {agent.exploration:.3f}")
        
        # Evaluation
        if episode_cnt % args.test_log_interval == 0:
            eval_res_dict = run_eval_episode(env, agent, args)
            print(f"[EVAL] Episode: {episode_cnt:4d}, "
                  f"Eval Reward: {eval_res_dict['episode_reward']:.2f}, "
                  f"Eval Win Rate: {eval_res_dict['win_rate']:.2f}")
    
    print("\n" + "=" * 60)
    print("Training Completed!")
    print("=" * 60)
    
    return agent


# Example usage
if __name__ == "__main__":
    """
    Example: Define your custom environment and test QMIX
    
    Your custom environment should have:
    - num_agents: int
    - n_actions: int  
    - obs_dim: int
    - state_dim: int (can be None if no global state)
    - episode_limit: int
    - reset() method
    - step(actions) method
    - get_available_actions() method (optional)
    """
    
    # Example: Simple dummy environment for demonstration
    class DummyMultiAgentEnv:
        def __init__(self):
            self.num_agents = 3
            self.n_actions = 5
            self.obs_dim = 10
            self.state_dim = 15
            self.episode_limit = 50
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
            reward = np.random.randn() * 0.1
            terminated = self._step_count >= self.episode_limit
            truncated = False
            info = {'battle_won': terminated and (np.random.rand() > 0.5)}
            return next_obs, next_state, reward, terminated, truncated, info
        
        def get_available_actions(self):
            return np.ones((self.num_agents, self.n_actions), dtype=np.int8)
    
    # Create your custom environment
    custom_env = DummyMultiAgentEnv()
    
    # Test QMIX on your custom environment
    trained_agent = test_qmix_with_custom_env(custom_env)
    
    print("\nYou can now use 'trained_agent' for inference or save it.")
    print("To use with your own environment, replace DummyMultiAgentEnv with your environment class.")
