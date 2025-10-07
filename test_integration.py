"""
Integration test for QMIX with custom environment

This script tests the complete flow from environment creation to QMIX training.
It uses a very short training run for quick verification.
"""

import sys
import numpy as np

def test_environment_interface():
    """Test that custom environment follows the required interface"""
    print("Testing environment interface...")
    
    from example_custom_env import CooperativeNavigationEnv
    
    env = CooperativeNavigationEnv(num_agents=2, episode_limit=10)
    
    # Check required attributes
    assert hasattr(env, 'num_agents'), "Missing num_agents"
    assert hasattr(env, 'n_actions'), "Missing n_actions"
    assert hasattr(env, 'obs_dim'), "Missing obs_dim"
    assert hasattr(env, 'state_dim'), "Missing state_dim"
    assert hasattr(env, 'episode_limit'), "Missing episode_limit"
    
    # Check reset method
    obs, state, info = env.reset()
    assert obs.shape == (env.num_agents, env.obs_dim), f"Wrong obs shape: {obs.shape}"
    assert state.shape == (env.state_dim,), f"Wrong state shape: {state.shape}"
    assert isinstance(info, dict), "Info should be a dictionary"
    
    # Check step method
    actions = np.random.randint(0, env.n_actions, size=env.num_agents)
    next_obs, next_state, reward, terminated, truncated, info = env.step(actions)
    assert next_obs.shape == (env.num_agents, env.obs_dim), f"Wrong next_obs shape"
    assert next_state.shape == (env.state_dim,), f"Wrong next_state shape"
    assert isinstance(reward, (int, float)), "Reward should be scalar"
    assert isinstance(terminated, bool), "Terminated should be bool"
    assert isinstance(truncated, bool), "Truncated should be bool"
    
    # Check available actions
    avail_actions = env.get_available_actions()
    assert avail_actions.shape == (env.num_agents, env.n_actions), f"Wrong avail_actions shape"
    
    print("✓ Environment interface test passed!")
    return True


def test_wrapper():
    """Test the environment wrapper"""
    print("\nTesting environment wrapper...")
    
    from example_custom_env import CooperativeNavigationEnv
    from test_qmix_custom_env import CustomEnvWrapper
    
    env = CooperativeNavigationEnv(num_agents=2, episode_limit=10)
    wrapped_env = CustomEnvWrapper(env)
    
    # Check wrapper attributes
    assert wrapped_env.num_agents == env.num_agents
    assert wrapped_env.n_actions == env.n_actions
    assert wrapped_env.obs_dim == env.obs_dim
    assert wrapped_env.state_dim == env.state_dim
    assert wrapped_env.episode_limit == env.episode_limit
    
    # Check wrapper shapes
    assert wrapped_env.obs_shape == (env.obs_dim,)
    assert wrapped_env.state_shape == (env.state_dim,)
    
    # Test reset and step through wrapper
    obs, state, info = wrapped_env.reset()
    actions = np.random.randint(0, env.n_actions, size=env.num_agents)
    next_obs, next_state, reward, terminated, truncated, info = wrapped_env.step(actions)
    
    print("✓ Wrapper test passed!")
    return True


def test_qmix_components():
    """Test QMIX agent and buffer creation"""
    print("\nTesting QMIX components...")
    
    import argparse
    import torch
    from example_custom_env import CooperativeNavigationEnv
    from test_qmix_custom_env import CustomEnvWrapper, create_qmix_agent
    from marltoolkit.data import ReplayBuffer
    
    # Create environment
    env = CooperativeNavigationEnv(num_agents=2, episode_limit=10)
    wrapped_env = CustomEnvWrapper(env)
    
    # Create args
    args = argparse.Namespace(
        num_agents=wrapped_env.num_agents,
        n_actions=wrapped_env.n_actions,
        obs_dim=wrapped_env.obs_dim,
        state_dim=wrapped_env.state_dim,
        episode_limit=wrapped_env.episode_limit,
        obs_shape=wrapped_env.get_actor_input_shape(),
        state_shape=wrapped_env.state_shape,
        action_shape=wrapped_env.action_shape,
        reward_shape=wrapped_env.reward_shape,
        done_shape=wrapped_env.done_shape,
        actor_input_dim=wrapped_env.obs_dim,
        fc_hidden_dim=32,
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
        target_update_interval=10,
        learner_update_freq=1,
        double_q=True,
        clip_grad_norm=10,
        replay_buffer_size=100,
        batch_size=8,
        memory_warmup_size=8,
        total_steps=100,
        train_log_interval=5,
        test_log_interval=20,
        num_eval_episodes=1,
        device='cpu',
    )
    
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
    
    # Create agent
    agent = create_qmix_agent(wrapped_env, args)
    
    # Test agent methods
    agent.init_hidden_states(batch_size=1)
    obs = np.random.randn(args.num_agents, args.obs_dim).astype(np.float32)
    avail_actions = np.ones((args.num_agents, args.n_actions), dtype=np.int8)
    
    # Test sampling
    actions = agent.sample(obs, avail_actions)
    assert actions.shape == (args.num_agents,), f"Wrong actions shape: {actions.shape}"
    
    # Test prediction
    actions = agent.predict(obs, avail_actions)
    assert actions.shape == (args.num_agents,), f"Wrong actions shape: {actions.shape}"
    
    print("✓ QMIX components test passed!")
    return True


def test_training_loop():
    """Test a very short training loop"""
    print("\nTesting training loop (short run)...")
    
    import argparse
    from example_custom_env import CooperativeNavigationEnv
    from test_qmix_custom_env import (
        CustomEnvWrapper, create_qmix_agent, 
        run_train_episode, run_eval_episode
    )
    from marltoolkit.data import ReplayBuffer
    
    # Create environment
    env = CooperativeNavigationEnv(num_agents=2, episode_limit=10)
    wrapped_env = CustomEnvWrapper(env)
    
    # Minimal config for fast test
    args = argparse.Namespace(
        num_agents=wrapped_env.num_agents,
        n_actions=wrapped_env.n_actions,
        obs_dim=wrapped_env.obs_dim,
        state_dim=wrapped_env.state_dim,
        episode_limit=wrapped_env.episode_limit,
        obs_shape=wrapped_env.get_actor_input_shape(),
        state_shape=wrapped_env.state_shape,
        action_shape=wrapped_env.action_shape,
        reward_shape=wrapped_env.reward_shape,
        done_shape=wrapped_env.done_shape,
        actor_input_dim=wrapped_env.obs_dim,
        fc_hidden_dim=16,
        rnn_hidden_dim=16,
        mixing_embed_dim=8,
        hypernet_layers=1,
        hypernet_embed_dim=16,
        gamma=0.99,
        learning_rate=0.001,
        min_learning_rate=0.0001,
        egreedy_exploration=0.5,
        min_exploration=0.1,
        target_update_tau=0.05,
        target_update_interval=5,
        learner_update_freq=1,
        double_q=True,
        clip_grad_norm=10,
        replay_buffer_size=50,
        batch_size=4,
        memory_warmup_size=4,
        total_steps=50,
        train_log_interval=10,
        test_log_interval=20,
        num_eval_episodes=1,
        device='cpu',
    )
    
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
    
    agent = create_qmix_agent(wrapped_env, args)
    
    # Warmup
    print("  Running warmup episodes...")
    while rpm.size() < args.memory_warmup_size:
        run_train_episode(wrapped_env, agent, rpm, args)
    
    # Short training
    print("  Running training episodes...")
    steps_cnt = 0
    episode_cnt = 0
    
    while steps_cnt < args.total_steps:
        train_res = run_train_episode(wrapped_env, agent, rpm, args)
        episode_cnt += 1
        steps_cnt += train_res['episode_step']
        
        agent.learning_rate = max(
            agent.lr_scheduler.step(train_res['episode_step']),
            agent.min_learning_rate,
        )
    
    # Test evaluation
    print("  Running evaluation...")
    eval_res = run_eval_episode(wrapped_env, agent, args)
    
    print(f"  Final episode: {episode_cnt}, Steps: {steps_cnt}")
    print(f"  Eval reward: {eval_res['episode_reward']:.2f}")
    
    print("✓ Training loop test passed!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("QMIX Integration Tests")
    print("="*60)
    
    tests = [
        ("Environment Interface", test_environment_interface),
        ("Environment Wrapper", test_wrapper),
        ("QMIX Components", test_qmix_components),
        ("Training Loop", test_training_loop),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print(f"{'='*60}")
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ {test_name} FAILED!")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {failed} test(s) failed!")
        sys.exit(1)
