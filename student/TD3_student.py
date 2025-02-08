import torch
import torch.optim as optim
import numpy as np
from nn_net.net import Actor, Critic
import copy
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TD3_student:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_target_2 = copy.deepcopy(self.critic_2)
        self.critic_optimizer = optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()),
            lr=3e-4
        )
        self.max_action = max_action
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, transition_dict, tau=0.005):
        self.total_it += 1
        state = torch.FloatTensor(transition_dict['states']).to(device)
        action = torch.FloatTensor(transition_dict['actions']).to(device)
        next_state = torch.FloatTensor(transition_dict['next_states']).to(device)
        reward = torch.FloatTensor(transition_dict['rewards']).to(device)
        done = torch.FloatTensor(transition_dict['dones']).to(device)

        with torch.no_grad():
            # 目标动作 + 噪声
            noise = (torch.randn_like(action) * 0.2).clamp(-0.5, 0.5)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)

            # 目标 Q 值
            target_q1 = self.critic_target_1(next_state, next_action)
            target_q2 = self.critic_target_2(next_state, next_action)
            target_q = reward + (1 - done) * 0.99 * torch.min(target_q1, target_q2)

        # 更新 Critic 网络
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        critic_loss1 = nn.MSELoss()(current_q1, target_q)
        critic_loss2 = nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        (critic_loss1 + critic_loss2).backward()
        self.critic_optimizer.step()

        # 延迟策略更新
        if self.total_it % 2 == 0:
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


    def save(self, filename1, filename2, filename3):
        torch.save(self.actor.state_dict(), filename1)
        torch.save(self.critic_1.state_dict(), filename2)
        torch.save(self.critic_2.state_dict(), filename3)

    def load(self, filename1, filename2, filename3):
        self.actor.load_state_dict(torch.load(filename1))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_1.load_state_dict(torch.load(filename2))
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_2.load_state_dict(torch.load(filename3))
        self.critic_target_2 = copy.deepcopy(self.critic_2)