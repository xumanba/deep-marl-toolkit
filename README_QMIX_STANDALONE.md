# QMIX 算法独立测试指南

本文档说明如何使用从 deep-marl-toolkit 仓库中抽离出的 QMIX 算法，在自定义环境中进行测试。

## 概述

QMIX 是一种值分解多智能体强化学习算法，适用于协作式多智能体任务。本仓库提供了一个独立的测试脚本 `test_qmix_custom_env.py`，可以方便地在自定义的离散动作空间环境中测试 QMIX 算法。

## 核心文件

- **test_qmix_custom_env.py**: 主测试脚本，包含完整的 QMIX 训练和评估流程
- **test_qmix_quick.py**: 快速测试脚本，用于验证设置
- **test_qmix_simple.py**: 基础测试脚本，验证组件初始化

## 环境要求

您的自定义环境需要满足以下要求：

### 必需属性

```python
class CustomEnvironment:
    num_agents: int        # 智能体数量
    n_actions: int         # 每个智能体的离散动作数
    obs_dim: int          # 观测维度
    state_dim: int        # 全局状态维度（可选，可为 None）
    episode_limit: int    # 每个回合的最大步数
```

### 必需方法

```python
def reset(self) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    重置环境
    
    Returns:
        obs: 观测数组，形状为 (num_agents, obs_dim)
        state: 全局状态数组，形状为 (state_dim,)
        info: 信息字典
    """
    pass

def step(self, actions: np.ndarray) -> Tuple:
    """
    执行一步
    
    Args:
        actions: 动作数组，形状为 (num_agents,)
        
    Returns:
        next_obs: 下一步观测，形状为 (num_agents, obs_dim)
        next_state: 下一步全局状态，形状为 (state_dim,)
        reward: 奖励标量
        terminated: 是否终止（bool）
        truncated: 是否截断（bool）
        info: 信息字典
    """
    pass

def get_available_actions(self) -> np.ndarray:
    """
    获取每个智能体的可用动作（可选）
    
    Returns:
        available_actions: 可用动作掩码，形状为 (num_agents, n_actions)
                          1 表示可用，0 表示不可用
    """
    # 如果未实现，将假定所有动作都可用
    pass
```

## 使用方法

### 方法 1: 使用默认配置

```python
from test_qmix_custom_env import test_qmix_with_custom_env

# 创建您的自定义环境
custom_env = YourCustomEnvironment()

# 使用默认配置测试 QMIX
trained_agent = test_qmix_with_custom_env(custom_env)
```

### 方法 2: 自定义配置

```python
import argparse
import torch
from test_qmix_custom_env import CustomEnvWrapper, create_qmix_agent, run_train_episode
from marltoolkit.data import ReplayBuffer

# 1. 包装您的环境
env = CustomEnvWrapper(your_custom_env)

# 2. 配置参数
args = argparse.Namespace(
    # 环境参数（从环境获取）
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
    actor_input_dim=env.obs_dim,  # 必需：actor 网络输入维度
    
    # 网络架构参数
    fc_hidden_dim=64,          # 全连接层隐藏维度
    rnn_hidden_dim=64,         # RNN 隐藏层维度
    mixing_embed_dim=32,       # Mixer 网络嵌入维度
    hypernet_layers=2,         # Hypernetwork 层数
    hypernet_embed_dim=64,     # Hypernetwork 嵌入维度
    
    # 训练参数
    gamma=0.99,                # 折扣因子
    learning_rate=0.0005,      # 学习率
    min_learning_rate=0.00001, # 最小学习率
    egreedy_exploration=1.0,   # 初始 epsilon
    min_exploration=0.01,      # 最小 epsilon
    target_update_tau=0.05,    # 目标网络软更新系数
    target_update_interval=100,# 目标网络更新间隔
    learner_update_freq=3,     # 学习器更新频率
    double_q=True,             # 使用 Double Q-learning
    clip_grad_norm=10,         # 梯度裁剪
    
    # 经验回放参数
    replay_buffer_size=5000,   # 回放缓冲区大小
    batch_size=32,             # 批次大小
    memory_warmup_size=32,     # 预热大小
    
    # 训练计划
    total_steps=50000,         # 总训练步数
    train_log_interval=20,     # 训练日志间隔
    test_log_interval=100,     # 测试日志间隔
    num_eval_episodes=5,       # 评估回合数
    
    # 设备
    device='cuda' if torch.cuda.is_available() else 'cpu',
)

# 3. 创建组件
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

# 4. 训练循环
# 详见 test_qmix_custom_env.py 中的完整示例
```

## 示例：简单多智能体环境

```python
import numpy as np

class SimpleMultiAgentEnv:
    """一个简单的示例环境"""
    
    def __init__(self):
        self.num_agents = 3        # 3 个智能体
        self.n_actions = 5         # 每个智能体 5 个动作
        self.obs_dim = 10          # 10 维观测
        self.state_dim = 15        # 15 维全局状态
        self.episode_limit = 50    # 每回合最多 50 步
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
        
        # 计算奖励（示例：随机奖励）
        reward = np.random.randn() * 0.1
        
        # 判断是否结束
        terminated = self._step_count >= self.episode_limit
        truncated = False
        
        # 可选：添加胜利信息
        info = {'battle_won': terminated and (np.random.rand() > 0.5)}
        
        return next_obs, next_state, reward, terminated, truncated, info
    
    def get_available_actions(self):
        # 返回所有动作都可用的掩码
        return np.ones((self.num_agents, self.n_actions), dtype=np.int8)

# 使用环境
if __name__ == "__main__":
    from test_qmix_custom_env import test_qmix_with_custom_env
    
    env = SimpleMultiAgentEnv()
    agent = test_qmix_with_custom_env(env)
```

## 快速测试

运行快速测试以验证设置：

```bash
# 测试基本组件
python test_qmix_simple.py

# 快速训练测试（短时间）
python test_qmix_quick.py

# 完整训练测试
python test_qmix_custom_env.py
```

## QMIX 算法说明

QMIX 算法的核心组件：

1. **RNN Actor Model**: 每个智能体的 Q 网络，使用 GRU 单元处理部分可观测性
2. **Mixer Model**: 将各智能体的 Q 值组合成全局 Q 值，使用 Hypernetwork 保证单调性
3. **经验回放**: 存储完整回合数据用于训练
4. **Epsilon-greedy 探索**: 在训练过程中逐渐减小探索率

## 代码结构

本实现严格遵循 marltoolkit 仓库中的 QMIX 实现模式：

- 使用 `marltoolkit.agents.qmix_agent.QMixAgent`
- 使用 `marltoolkit.modules.actors.RNNActorModel` 作为 actor 网络
- 使用 `marltoolkit.modules.mixers.QMixerModel` 作为 mixer 网络
- 使用 `marltoolkit.data.ReplayBuffer` 进行经验回放
- 遵循 `scripts/main_qmix.py` 的训练流程

## 注意事项

1. **离散动作空间**: QMIX 仅适用于离散动作空间
2. **协作任务**: QMIX 设计用于完全协作的多智能体任务
3. **全局奖励**: 环境应返回一个共享的团队奖励
4. **部分可观测**: QMIX 可以处理部分可观测环境（通过 RNN）
5. **内存需求**: 回放缓冲区会存储完整回合，注意内存使用

## 保存和加载模型

```python
# 保存模型
agent.save_model(save_dir='./models', 
                 actor_model_name='actor.th',
                 mixer_model_name='mixer.th',
                 opt_name='optimizer.th')

# 加载模型
agent.load_model(save_dir='./models',
                 actor_model_name='actor.th',
                 mixer_model_name='mixer.th',
                 opt_name='optimizer.th')
```

## 故障排除

### 问题：`AttributeError: 'RNNActorModel' object has no attribute 'actor_input_dim'`

**解决方案**: 确保在创建 args 时设置了 `actor_input_dim` 属性：
```python
args.actor_input_dim = env.obs_dim
```

### 问题：训练不稳定或不收敛

**可能的解决方案**:
- 减小学习率
- 增加目标网络更新间隔
- 调整 epsilon 衰减速度
- 增大回放缓冲区大小
- 检查奖励设计是否合理

### 问题：内存不足

**解决方案**:
- 减小回放缓冲区大小
- 减小批次大小
- 缩短回合长度限制

## 参考

- 原始 QMIX 论文: [QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- 仓库代码: `marltoolkit/agents/qmix_agent.py`
- 示例脚本: `scripts/main_qmix.py`

## 许可证

本代码遵循原仓库的许可证。
