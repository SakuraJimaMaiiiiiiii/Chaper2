import numpy as np
import random
from collections import deque

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[dataIdx]

    def total(self):
        return self.tree[0]


class ReplayBuffer:
    def __init__(self, capacity,  use_per=False, use_her=False, k_goals=4,
                 per_alpha=0.6, per_beta=0.4, per_beta_increment=0.001, per_epsilon=0.01,
                 reward_scale=1.0):
     
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.use_per = use_per
        self.use_her = use_her
        self.k_goals = k_goals

        self.per_alpha = per_alpha
        self.per_beta = per_beta
        self.per_beta_increment = per_beta_increment
        self.per_epsilon = per_epsilon
        self.reward_scale = reward_scale

        if use_per:
            self.tree = SumTree(capacity)
            self.default_priority = 1.0
        else:
            self.default_priority = None

    def store(self, state, action, next_state, reward, done, goal=None):
        """存储单个transition"""
        reward *= self.reward_scale
        transition = (state, action, next_state, reward, done, goal)
        if not isinstance(transition, tuple) or len(transition) != 6:
            raise ValueError()

        if self.use_per:
            if self.tree.n_entries > 0:
                leaf_priorities = self.tree.tree[-self.tree.capacity: self.tree.capacity-1+self.tree.capacity]
                leaf_priorities = leaf_priorities[:self.tree.n_entries]
                if len(leaf_priorities) > 0:
                    max_priority = leaf_priorities.max()
                    max_priority = max(max_priority, self.per_epsilon)
                else:
                    max_priority = self.default_priority
            else:
                max_priority = self.default_priority

            self.tree.add(max_priority, transition)
        else:
            self.buffer.append(transition)

    def store_episode(self, episode_transitions):
        """
        存储整个episode的数据（列表形式）后，再根据HER增加额外经验。
        episode_transitions: [(s, a, s', r, d), ...]
        """
        if not self.use_her:
            # 无HER
            for (s, a, s_next, r, d) in episode_transitions:
                self.store(s, a, s_next, r, d, goal=None)
        else:
            length = len(episode_transitions)
            for i, (s, a, s_next, r, d) in enumerate(episode_transitions):
                # 原始transition存储
                self.store(s, a, s_next, r, d, goal=None)
                if not d:
                    her_goals = []
                    for _ in range(self.k_goals):
                        if i < length - 1:
                            future_idx = np.random.randint(i+1, length)
                            future_goal = episode_transitions[future_idx][0].copy() 
                        else:
                            future_goal = s_next.copy()
                        her_goals.append(future_goal)
                    for fg in her_goals:
                        new_reward = -np.linalg.norm(s_next - fg) * self.reward_scale
                        self.store(s, a, s_next, new_reward, d, goal=fg)

    def sample(self, batch_size):
        if self.use_per:
            if self.tree.n_entries == 0 or self.tree.total() == 0:
                return [], None, None

            batch = []
            indices = []
            weights = np.zeros(batch_size)           # 采样性权重
            segment = self.tree.total() / batch_size
            self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

            total_p = self.tree.total()
            if total_p == 0:
                return [], None, None

            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                s = random.uniform(a, b)
                idx, p, data = self.tree.get(s)
              
                if not isinstance(data, tuple):
                    continue
                batch.append(data)
                indices.append(idx)

                prob = p / (total_p + 1e-5)
               
                if prob <= 0:
                    prob = self.per_epsilon
                weights[i] = (prob * self.tree.n_entries) ** (-self.per_beta)     #   （pi^a/Σpk^a）^-β

            if len(batch) == 0:
                return [], None, None

            weights /= weights.max() if weights.max() != 0 else 1.0
            return batch, indices, weights
        else:
            if len(self.buffer) < batch_size:
                return [], None, None
            batch = random.sample(self.buffer, batch_size)
            return batch, None, None

    def update_priorities(self, indices, priorities):
        """更新优先级"""
        if self.use_per and indices is not None and priorities is not None:
            priorities = np.array(priorities).flatten() 
            for idx, priority in zip(indices, priorities):
                priority = np.clip(priority, self.per_epsilon, 1.0)
                priority = float(priority ** self.per_alpha)  
                if isinstance(priority, np.ndarray):
                    priority = float(priority.item()) 
                self.tree.update(idx, priority)

    def __len__(self):
        if self.use_per:
            return self.tree.n_entries
        return len(self.buffer)