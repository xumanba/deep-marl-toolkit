# QMIX 快速开始指南 / QMIX Quick Start Guide

## 🚀 一分钟开始 / Get Started in 1 Minute

### 中文

```python
# 1. 定义你的环境
class MyEnv:
    def __init__(self):
        self.num_agents = 3      # 智能体数量
        self.n_actions = 5       # 动作数量
        self.obs_dim = 10        # 观测维度
        self.state_dim = 15      # 状态维度
        self.episode_limit = 50  # 最大步数
    
    def reset(self):
        obs = ...  # (num_agents, obs_dim)
        state = ...  # (state_dim,)
        return obs, state, {}
    
    def step(self, actions):
        next_obs, next_state = ...
        reward = ...  # 标量
        terminated = ...  # bool
        truncated = False
        return next_obs, next_state, reward, terminated, truncated, {}
    
    def get_available_actions(self):
        return np.ones((self.num_agents, self.n_actions))  # 可选

# 2. 训练 QMIX
from test_qmix_custom_env import test_qmix_with_custom_env

env = MyEnv()
agent = test_qmix_with_custom_env(env)

# 3. 使用训练好的智能体
obs, state, _ = env.reset()
actions = agent.predict(obs, env.get_available_actions())
```

### English

```python
# 1. Define your environment
class MyEnv:
    def __init__(self):
        self.num_agents = 3      # Number of agents
        self.n_actions = 5       # Number of actions
        self.obs_dim = 10        # Observation dimension
        self.state_dim = 15      # State dimension
        self.episode_limit = 50  # Max steps
    
    def reset(self):
        obs = ...  # (num_agents, obs_dim)
        state = ...  # (state_dim,)
        return obs, state, {}
    
    def step(self, actions):
        next_obs, next_state = ...
        reward = ...  # scalar
        terminated = ...  # bool
        truncated = False
        return next_obs, next_state, reward, terminated, truncated, {}
    
    def get_available_actions(self):
        return np.ones((self.num_agents, self.n_actions))  # optional

# 2. Train QMIX
from test_qmix_custom_env import test_qmix_with_custom_env

env = MyEnv()
agent = test_qmix_with_custom_env(env)

# 3. Use trained agent
obs, state, _ = env.reset()
actions = agent.predict(obs, env.get_available_actions())
```

## 📚 文档 / Documentation

| 文档 / Document | 说明 / Description |
|----------------|-------------------|
| **QMIX_EXTRACTION_SUMMARY.md** | 完整总结 / Complete summary |
| **QMIX_USAGE.md** | 英文详细文档 / English detailed guide |
| **README_QMIX_STANDALONE.md** | 中文详细文档 / Chinese detailed guide |

## 🧪 测试 / Tests

```bash
# 基础测试 / Basic test
python test_qmix_simple.py

# 快速训练测试 / Quick training test  
python test_qmix_quick.py

# 完整集成测试 / Full integration test
python test_integration.py

# 示例环境 / Example environment
python example_custom_env.py
```

## 📁 核心文件 / Core Files

- **test_qmix_custom_env.py** - 主脚本 / Main script
- **example_custom_env.py** - 示例环境 / Example environment
- **test_integration.py** - 集成测试 / Integration tests

## ✅ 环境要求 / Environment Requirements

你的环境需要实现 / Your environment needs to implement:

| 属性 / Attribute | 类型 / Type | 说明 / Description |
|-----------------|-------------|-------------------|
| `num_agents` | int | 智能体数量 / Number of agents |
| `n_actions` | int | 动作数量 / Number of actions |
| `obs_dim` | int | 观测维度 / Observation dimension |
| `state_dim` | int | 状态维度 / State dimension (can be None) |
| `episode_limit` | int | 最大步数 / Max steps per episode |

| 方法 / Method | 返回 / Returns | 说明 / Description |
|--------------|---------------|-------------------|
| `reset()` | (obs, state, info) | 重置环境 / Reset environment |
| `step(actions)` | (obs, state, reward, term, trunc, info) | 执行动作 / Execute actions |
| `get_available_actions()` | action_mask | 可用动作 / Available actions (optional) |

## 🎯 示例环境 / Example Environment

查看 `example_custom_env.py` 中的 `CooperativeNavigationEnv` 类，这是一个完整的示例。

See the `CooperativeNavigationEnv` class in `example_custom_env.py` for a complete example.

## 🔧 自定义配置 / Custom Configuration

如需自定义超参数，参见完整文档：

For custom hyperparameters, see full documentation:

```python
import argparse
from test_qmix_custom_env import CustomEnvWrapper, create_qmix_agent

env = CustomEnvWrapper(your_env)
args = argparse.Namespace(
    # ... 自定义配置 / custom configuration
    learning_rate=0.001,
    gamma=0.99,
    # ... 更多参数见文档 / see docs for more
)
agent = create_qmix_agent(env, args)
```

## ⚠️ 重要提示 / Important Notes

1. **离散动作空间** / **Discrete Actions Only**: QMIX 仅支持离散动作
2. **协作任务** / **Cooperative Tasks**: 适用于完全协作任务
3. **共享奖励** / **Shared Reward**: 环境应返回单一的团队奖励
4. **遵循仓库模式** / **Follows Repo Pattern**: 严格遵循原仓库实现

## 🐛 问题排查 / Troubleshooting

### 错误 / Error: `actor_input_dim` not found

```python
# 解决方案 / Solution:
args.actor_input_dim = env.obs_dim
```

### 训练不稳定 / Training Unstable

- 降低学习率 / Reduce learning rate
- 增大回放缓冲区 / Increase replay buffer
- 调整探索率 / Adjust exploration rate

## 📞 支持 / Support

1. 查看文档 / Check documentation
2. 运行测试 / Run tests: `python test_integration.py`
3. 查看示例 / See example: `example_custom_env.py`

## 📊 测试状态 / Test Status

✅ All tests passing (4/4)

```
✓ Environment Interface Test
✓ Environment Wrapper Test
✓ QMIX Components Test
✓ Training Loop Test
```

---

**开始使用 / Get Started**: 运行 `python example_custom_env.py`  
**查看文档 / See Docs**: 打开 `QMIX_USAGE.md` 或 `README_QMIX_STANDALONE.md`  
**运行测试 / Run Tests**: 执行 `python test_integration.py`
