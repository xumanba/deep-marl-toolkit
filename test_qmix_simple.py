"""
Simple test to verify the QMIX setup
"""
import sys
import torch
import numpy as np

# Test imports
try:
    from marltoolkit.agents.qmix_agent import QMixAgent
    from marltoolkit.data import ReplayBuffer
    from marltoolkit.modules.actors import RNNActorModel
    from marltoolkit.modules.mixers import QMixerModel
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test basic initialization
try:
    import argparse
    
    args = argparse.Namespace(
        num_agents=3,
        n_actions=5,
        obs_dim=10,
        state_dim=15,
        episode_limit=50,
        obs_shape=(10,),
        state_shape=(15,),
        action_shape=(5,),
        reward_shape=(1,),
        done_shape=(1,),
        fc_hidden_dim=64,
        rnn_hidden_dim=64,
        mixing_embed_dim=32,
        hypernet_layers=2,
        hypernet_embed_dim=64,
        actor_input_dim=10,  # Same as obs_dim
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
        replay_buffer_size=1000,
        batch_size=32,
        memory_warmup_size=32,
        total_steps=1000,
        device='cpu',
    )
    
    # Create models
    # Note: RNNActorModel expects actor_input_dim to be set on args
    # This should match the input dimension for the actor (observation dimension)
    actor_model = RNNActorModel(args)
    mixer_model = QMixerModel(
        num_agents=args.num_agents,
        state_dim=args.state_dim,
        mixing_embed_dim=args.mixing_embed_dim,
        hypernet_layers=args.hypernet_layers,
        hypernet_embed_dim=args.hypernet_embed_dim,
    )
    
    # Create agent
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
    
    print("✓ QMIX agent created successfully")
    
    # Test agent methods
    agent.init_hidden_states(batch_size=1)
    obs = np.random.randn(args.num_agents, args.obs_dim).astype(np.float32)
    available_actions = np.ones((args.num_agents, args.n_actions), dtype=np.int8)
    
    # Test sampling
    actions = agent.sample(obs, available_actions)
    print(f"✓ Sampled actions: {actions}")
    
    # Test prediction
    actions = agent.predict(obs, available_actions)
    print(f"✓ Predicted actions: {actions}")
    
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
    print("✓ Replay buffer created successfully")
    
    print("\n" + "="*50)
    print("All tests passed! QMIX setup is working correctly.")
    print("="*50)
    
except Exception as e:
    print(f"✗ Error during initialization: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
