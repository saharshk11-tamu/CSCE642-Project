from Solvers.AbstractSolver import AbstractSolver

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from collections import deque
import random

class QFunction(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[64, 64]):
        super().__init__()
        self.state_size = state_size
        size = [state_size] + hidden_sizes + [action_size]
        self.layers = nn.ModuleList()
        for i in range(len(size) - 1):
            self.layers.append(nn.Linear(size[i], size[i+1]))
    
    def forward(self, obs):
        if not torch.is_tensor(obs):
            x = torch.as_tensor(obs, dtype=torch.float32)
        else:
            x = obs.float()

        if x.ndim == 1 and x.shape[0] != self.state_size:
            raise ValueError('Expected flattened state representation')

        for i in range(len(self.layers)-1):
            x = torch.relu(self.layers[i](x))
        return self.layers[-1](x)

class DQN(AbstractSolver):
    def __init__(
            self,
            env,
            epsilon = 0.1,
            gamma = 0.1,
            num_episodes = 100,
            max_steps = 100,
            layers=[64, 64],
            replay_buffer_size=10000,
            batch_size=64,
            update_target_every=10
        ):
        super().__init__(env, epsilon, gamma, num_episodes, max_steps)
        self.model = QFunction(env.observation_space.n, env.action_space.n, layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.target_model = deepcopy(self.model)
        self.loss_fn = nn.SmoothL1Loss()

        for p in self.target_model.parameters():
            p.requires_grad = False

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.update_target_every = update_target_every

        self.n_steps = 0

    def _encode_state(self, state):
        state_tensor = torch.as_tensor(state)
        if state_tensor.ndim == 0:
            state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.long()
        encoded = F.one_hot(state_tensor, num_classes=self.env.observation_space.n).float()
        if encoded.shape[0] == 1:
            return encoded.squeeze(0)
        return encoded

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def epsilon_greedy_policy(self, state):
        probs = np.ones(self.env.action_space.n)
        state_tensor = self._encode_state(state)
        q_values = self.model(state_tensor)
        if q_values.ndim > 1:
            q_values = q_values.squeeze(0)
        A_star = torch.argmax(q_values).item()
        probs *= self.epsilon / self.env.action_space.n
        probs[A_star] += 1.0 - self.epsilon
        return probs

    def compute_target_values(self, next_states, rewards, dones):
        targets = []
        for i in range(len(rewards)):
            y = rewards[i]
            if not dones[i]:
                y += self.gamma * torch.max(self.target_model(next_states[i]))
            targets.append(y)
        
        return torch.as_tensor(targets)
    
    def replay(self):
        if len(self.replay_buffer) > self.batch_size:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
            minibatch = [
                np.array([
                    transition[idx]
                    for transition, idx in zip(minibatch, [i]*len(minibatch))
                ])
                for i in range(5)
            ]

            states, actions, rewards, next_states, dones = minibatch
            states = self._encode_state(states)
            actions = torch.as_tensor(actions, dtype=torch.float32)
            rewards = torch.as_tensor(rewards, dtype=torch.float32)
            next_states = self._encode_state(next_states)
            dones = torch.as_tensor(dones, dtype=torch.float32)

            current_q = self.model(states)
            current_q = torch.gather(
                current_q, dim=1, index=actions.unsqueeze(1).long()
            ).squeeze(-1)

            with torch.no_grad():
                target_q = self.compute_target_values(next_states, rewards, dones)

            loss_q = self.loss_fn(current_q, target_q)

            # Optimize the Q-network
            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()
        
    def memorize(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_episode(self):
        state, _ = self.env.reset()
        self.reward = 0
        for _ in range(self.max_steps):
            action = np.random.choice(np.arange(self.env.action_space.n), p=self.epsilon_greedy_policy(state))
            _, next_state, reward, done, _ = self.env.step(action)
            self.reward += reward
            self.memorize(state, action, reward, next_state, done)
            self.replay()

            self.n_steps += 1
            if self.n_steps == self.update_target_every:
                self.update_target_model()
                self.n_steps = 0
            state = next_state
            if done: break
    
    def create_greedy_policy(self):
        def policy_fn(state):
            state_tensor = self._encode_state(state)
            q_values = self.model(state_tensor)
            if q_values.ndim > 1:
                q_values = q_values.squeeze(0)
            return torch.argmax(q_values).detach().item()
        return policy_fn
