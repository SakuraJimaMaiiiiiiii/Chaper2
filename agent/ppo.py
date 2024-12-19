import torch
import torch.optim as optim
import numpy as np
from nn_net.net import ActorCritic


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


