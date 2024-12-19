import torch
import torch.optim as optim
import numpy as np
from nn_net.net import ActorCritic, Actor, Critic
import copy
from Buffer.replaybuffer import ReplayBuffer
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO
class PPO:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        self.ac = ActorCritic(obs_dim, act_dim).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=config.lr)

    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(device)
        with torch.no_grad():
            mean, std, _ = self.ac(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        return action.cpu().numpy(), log_prob.cpu().numpy()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        next_value = values[-1]
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.config.gamma * next_value * mask - values[step]
            gae = delta + self.config.gamma * self.config.lam * gae * mask
            advantages.insert(0, gae)
            next_value = values[step]
        return advantages

    def update(self, obs_buf, act_buf, adv_buf, ret_buf, logp_buf):
        obs_buf = torch.tensor(obs_buf, dtype=torch.float32).to(device)
        act_buf = torch.tensor(act_buf, dtype=torch.float32).to(device)
        adv_buf = torch.tensor(adv_buf, dtype=torch.float32).to(device)
        ret_buf = torch.tensor(ret_buf, dtype=torch.float32).to(device)
        logp_buf = torch.tensor(logp_buf, dtype=torch.float32).to(device)

        dataset_size = obs_buf.shape[0]
        total_approx_kl = 0.0  

        for _ in range(self.config.train_iters):
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.config.minibatch_size):
                end = start + self.config.minibatch_size
                minibatch_indices = indices[start:end]

                obs_batch = obs_buf[minibatch_indices]
                act_batch = act_buf[minibatch_indices]
                adv_batch = adv_buf[minibatch_indices]
                ret_batch = ret_buf[minibatch_indices]
                logp_old_batch = logp_buf[minibatch_indices]

                mean, std, value = self.ac(obs_batch)
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(act_batch).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1).mean()

                ratio = torch.exp(logp - logp_old_batch)

                # PPO目标函数
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * adv_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失
                value_loss = ((value - ret_batch) ** 2).mean()

                # 熵奖励权重
                loss = policy_loss + 0.5 * value_loss - 0.02 * entropy

                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 累加KL散度
                approx_kl = (logp_old_batch - logp).mean().item()
                total_approx_kl += approx_kl

            mean_approx_kl = total_approx_kl / (dataset_size / self.config.minibatch_size)
            if mean_approx_kl > 1.5 * self.config.target_kl:
                break

    def save_model(self, filepath):
        torch.save(self.ac.state_dict(), filepath)
        print(f"模型已保存")

    def load_model(self, filepath):
        self.ac.load_state_dict(torch.load(filepath, map_location=device))
        print(f"模型已加载")


class TD3:
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        max_action,
        use_per=False,
        use_her=False,
        buffer_size=1000000,
        k_goals=4,
        per_alpha=0.6,
        per_beta=0.4,
        per_beta_increment=0.001,
        per_epsilon=0.01,
        reward_scale=1.0
    ):
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
        self.use_per = use_per
        self.use_her = use_her
        
       
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_size,
            use_per=use_per,
            use_her=use_her,
            k_goals=k_goals,
            per_alpha=per_alpha,
            per_beta=per_beta,
            per_beta_increment=per_beta_increment,
            per_epsilon=per_epsilon,
            reward_scale=reward_scale
        )
        
        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.total_it += 1

   
        batch, indices, weights = self.replay_buffer.sample(batch_size)
        
    
        if self.use_per:
         
            state = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
            action = torch.FloatTensor(np.array([b[1] for b in batch])).to(device)
            next_state = torch.FloatTensor(np.array([b[2] for b in batch])).to(device)
            reward = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
            done = torch.FloatTensor(np.array([b[4] for b in batch])).to(device)
            weights = torch.FloatTensor(weights).to(device)
        else:
            state, action, next_state, reward, done, _ = zip(*batch)
            state = torch.FloatTensor(np.array(state)).to(device)
            action = torch.FloatTensor(np.array(action)).to(device)
            next_state = torch.FloatTensor(np.array(next_state)).to(device)
            reward = torch.FloatTensor(np.array(reward)).to(device)
            done = torch.FloatTensor(np.array(done)).to(device)
            weights = None

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
        current_q2 = self.critic_2(state, action)

        # 使用重要性采样权重
        if self.use_per:
            # 计算TD误差
            td_error1 = current_q1 - target_q
            td_error2 = current_q2 - target_q
            critic_loss1 = (weights * td_error1.pow(2)).mean()
            critic_loss2 = (weights * td_error2.pow(2)).mean()
            
            # 更新优先级  更新pre_memory
            with torch.no_grad():
                priorities = torch.abs(td_error1 + td_error2).cpu().numpy()
                self.replay_buffer.update_priorities(indices, priorities)
        else:
            critic_loss1 = nn.MSELoss()(current_q1, target_q)
            critic_loss2 = nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        (critic_loss1 + critic_loss2).backward()
        self.critic_optimizer.step()

        # 延迟策略更新
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

    def store_transition(self, state, action, next_state, reward, done, goal=None):
     
        self.replay_buffer.store(state, action, next_state, reward, done, goal)

    def store_episode(self, episode_transitions):
  
        self.replay_buffer.store_episode(episode_transitions)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "_actor.pth")
        torch.save(self.critic_1.state_dict(), filename + "_critic1.pth")
        torch.save(self.critic_2.state_dict(), filename + "_critic2.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "_actor.pth"))
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_1.load_state_dict(torch.load(filename + "_critic1.pth"))
        self.critic_target_1 = copy.deepcopy(self.critic_1)
        self.critic_2.load_state_dict(torch.load(filename + "_critic2.pth"))
        self.critic_target_2 = copy.deepcopy(self.critic_2)