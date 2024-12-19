import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from collections import deque
import random
from env.environment import Environment  # 自定义环境

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Actor 网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.layer_1(state))
        a = torch.relu(self.layer_2(a))
        return self.max_action * torch.tanh(self.layer_3(a))

# Critic 网络
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)

    def forward(self, state, action):
        q = torch.relu(self.layer_1(torch.cat([state, action], 1)))
        q = torch.relu(self.layer_2(q))
        return self.layer_3(q)


class TD3:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)
        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=3e-4)

        self.max_action = max_action
        self.replay_buffer = deque(maxlen=1000000)
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.total_it += 1

        # 从经验回放中采样
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, next_state, reward, done = zip(*batch)
        state = torch.FloatTensor(np.array(state)).to(device)
        action = torch.FloatTensor(np.array(action)).to(device)
        next_state = torch.FloatTensor(np.array(next_state)).to(device)
        reward = torch.FloatTensor(np.array(reward)).to(device)
        done = torch.FloatTensor(np.array(done)).to(device)

        with torch.no_grad():
            # 目标动作 + 噪声
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # 目标 Q 值
            target_q1 = self.critic_target_1(next_state, next_action)
            target_q2 = self.critic_target_2(next_state, next_action)
            target_q = reward + (1 - done) * discount * torch.min(target_q1, target_q2)

        # 更新 Critic 网络
        current_q1 = self.critic_1(state, action)
        loss_q1 = nn.MSELoss()(current_q1, target_q)
        current_q2 = self.critic_2(state, action)
        loss_q2 = nn.MSELoss()(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        (loss_q1 + loss_q2).backward()
        self.critic_optimizer.step()

        # 每隔 policy_freq 更新 Actor 网络
        if self.total_it % policy_freq == 0:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 软更新目标网络
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.critic_1.parameters(), self.critic_target_1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_target_2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def store_transition(self, state, action, next_state, reward, done):
        self.replay_buffer.append((state, action, next_state, reward, done))

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic_1.state_dict(), filename + "_critic1.pth")
        torch.save(self.critic_2.state_dict(), filename + "_critic2.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.critic_1.load_state_dict(torch.load(filename + "_critic1.pth"))
        self.critic_2.load_state_dict(torch.load(filename + "_critic2.pth"))

if __name__ == '__main__':
    # 训练过程
    env = Environment()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3(state_dim, action_dim, max_action)
    exploration_noise = 0.1
    episodes = 400  # 增加训练轮数
    batch_size = 100
    warmup_steps = 5000  # 减少随机探索步数
    steps = 0

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            # 初期探索
            if steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
                action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)
            
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, next_state, reward, done)
            state = next_state
            episode_reward += reward
            steps += 1

            # 更新 TD3 网络
            if len(agent.replay_buffer) > batch_size:
                agent.train(batch_size)

        # 动态调整探索噪声
        exploration_noise = max(0.01, exploration_noise * 0.995)

        print(f"Episode: {episode}, Reward: {episode_reward}")

        # 保存模型
        if episode % 50 == 0:
            agent.save("td3_agent")

    env.close()
