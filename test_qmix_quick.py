"""
Quick test of QMIX with custom environment
"""
import numpy as np
from test_qmix_custom_env import CustomEnvWrapper, test_qmix_with_custom_env

# Simple dummy environment for demonstration
class DummyMultiAgentEnv:
    def __init__(self):
        self.num_agents = 3
        self.n_actions = 5
        self.obs_dim = 10
        self.state_dim = 15
        self.episode_limit = 20  # Shorter for quick test
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

if __name__ == "__main__":
    import argparse
    from marltoolkit.agents.qmix_agent import QMixAgent
    from marltoolkit.data import ReplayBuffer
    from marltoolkit.modules.actors import RNNActorModel
    from marltoolkit.modules.mixers import QMixerModel
    import torch
    
    # Create custom environment
    custom_env = DummyMultiAgentEnv()
    env = CustomEnvWrapper(custom_env)
    
    # Quick configuration for fast testing
    args = argparse.Namespace(
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
        actor_input_dim=env.obs_dim,
        fc_hidden_dim=32,  # Smaller network
        rnn_hidden_dim=32,
        mixing_embed_dim=16,
        hypernet_layers=1,
        hypernet_embed_dim=32,
        gamma=0.99,
        learning_rate=0.001,
        min_learning_rate=0.0001,
        egreedy_exploration=1.0,
        min_exploration=0.1,
        target_update_tau=0.05,
        target_update_interval=50,
        learner_update_freq=2,
        double_q=True,
        clip_grad_norm=10,
        replay_buffer_size=200,
        batch_size=16,
        memory_warmup_size=16,
        total_steps=500,  # Very short for quick test
        train_log_interval=5,
        test_log_interval=20,
        num_eval_episodes=2,
        device='cpu',  # Use CPU for stability
    )
    
    print("="*60)
    print("Quick QMIX Test")
    print("="*60)
    print(f"Training for {args.total_steps} steps...")
    
    # Import runner functions
    from test_qmix_custom_env import run_train_episode, run_eval_episode, create_qmix_agent
    
    # Create components
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
    
    # Warmup
    print("\n[Warmup]")
    while rpm.size() < args.memory_warmup_size:
        run_train_episode(env, agent, rpm, args)
    print(f"Buffer size: {rpm.size()}")
    
    # Training
    print("\n[Training]")
    steps_cnt = 0
    episode_cnt = 0
    
    while steps_cnt < args.total_steps:
        train_res = run_train_episode(env, agent, rpm, args)
        episode_cnt += 1
        steps_cnt += train_res['episode_step']
        
        agent.learning_rate = max(
            agent.lr_scheduler.step(train_res['episode_step']),
            agent.min_learning_rate,
        )
        
        if episode_cnt % args.train_log_interval == 0:
            loss_str = f"{train_res['loss']:.4f}" if train_res['loss'] is not None else "N/A"
            print(f"Ep {episode_cnt:3d} | Steps {steps_cnt:4d} | "
                  f"Reward {train_res['episode_reward']:.2f} | Loss {loss_str}")
    
    # Final evaluation
    print("\n[Final Evaluation]")
    eval_res = run_eval_episode(env, agent, args)
    print(f"Eval Reward: {eval_res['episode_reward']:.2f}")
    print(f"Eval Win Rate: {eval_res['win_rate']:.2f}")
    
    print("\n" + "="*60)
    print("✓ Test completed successfully!")
    print("="*60)
