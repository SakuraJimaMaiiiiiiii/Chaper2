import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


#ppo网络结构
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.hidden_size = 128  # 隐藏层

        # 共享层
        self.shared_layers = nn.Sequential(
            nn.Linear(obs_dim, self.hidden_size),
            nn.ReLU(),
        )

        # 策略网络
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, act_dim),
        )

        # 价值网络
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

        # 动作的对数标准差（用于高斯策略）
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        shared = self.shared_layers(x)
        mean = self.actor(shared)          # [x,y,z]
        value = self.critic(shared).squeeze(-1)   # [int]
        std = torch.exp(self.log_std)      # [x,y,z]
        return mean, std, value


# BC Actor网络
class BCActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(BCActor, self).__init__()
        # 适中的隐藏层大小
        self.hidden_size1 = 256
        self.hidden_size2 = 256
        
        # 简单的三层网络结构
        self.layer_1 = nn.Linear(state_dim, self.hidden_size1)
        self.layer_2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.layer_3 = nn.Linear(self.hidden_size2, action_dim)
        
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer_1(state))
        # x = F.relu(self.layer_2(x))
        return self.max_action * torch.tanh(self.layer_3(x))
    
#TD3网络结构

# Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.hidden_size1 = 400
        self.hidden_size2 = 300
        self.layer_1 = nn.Linear(state_dim, self.hidden_size1)
        self.layer_2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.layer_3 = nn.Linear(self.hidden_size2, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.layer_1(state))
        a = torch.relu(self.layer_2(a))
        return self.max_action * torch.tanh(self.layer_3(a))

# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.hidden_size1 = 400
        self.hidden_size2 = 300
        self.layer_1 = nn.Linear(state_dim + action_dim, self.hidden_size1)
        self.layer_2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.layer_3 = nn.Linear(self.hidden_size2, 1)

    def forward(self, state, action):
        q = torch.relu(self.layer_1(torch.cat([state, action], 1)))
        q = torch.relu(self.layer_2(q))
        return self.layer_3(q)

class BCNet(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(BCNet, self).__init__()
        self.hidden_size1 = 256
        self.hidden_size2 = 128
        
        # 共享特征提取层
        self.layer_1 = nn.Linear(state_dim, self.hidden_size1)
        self.layer_2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        
        # 分别预测均值和标准差
        self.mean_layer = nn.Linear(self.hidden_size2, action_dim)
        self.log_std_layer = nn.Linear(self.hidden_size2, action_dim)
        
        # Dropout层
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        
        self.max_action = max_action
        self.min_log_std = -5
        self.max_log_std = 2
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in [self.layer_1, self.layer_2, self.mean_layer, self.log_std_layer]:
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, state):
        x = F.relu(self.layer_1(state))
        x = self.dropout1(x)
        x = F.relu(self.layer_2(x))
        x = self.dropout2(x)
        
        # 预测均值，确保更好的缩放
        mean = self.max_action * torch.tanh(self.mean_layer(x))
        
        # 调整标准差的预测
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        # 添加一个最小标准差
        std = torch.exp(log_std) + 1e-3
        
        return mean, std