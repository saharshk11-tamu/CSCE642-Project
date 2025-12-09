from Solvers.AbstractSolver import AbstractSolver

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from collections import deque
import random


class QFunction(nn.Module):
    def __init__(self, state_size, num_agents, hidden_sizes=[128, 128]):
        super().__init__()
        self.state_size = state_size
        self.num_agents = num_agents
        self.output_size = num_agents * 4  # 4 actions per agent
        size = [state_size] + hidden_sizes + [self.output_size]
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
        x = self.layers[-1](x)
        # reshape to (batch, num_agents, 4)
        if x.ndim == 1:
            x = x.view(self.num_agents, 4)
        else:
            x = x.view(x.shape[0], self.num_agents, 4)
        return x

class DQN(AbstractSolver):
    def __init__(
            self,
            env,
            epsilon = 0.1,
            gamma = 0.99,
            num_episodes = 100,
            max_steps = 100,
            layers=[64, 64],
            replay_buffer_size=10000,
            batch_size=64,
            update_target_every=10
        ):
        super().__init__(env, epsilon, gamma, num_episodes, max_steps)
        self.num_agents = env.num_agents
        self.state_size = self._calc_state_size()
        self.model = QFunction(self.state_size, self.num_agents, layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.target_model = deepcopy(self.model)
        self.loss_fn = nn.SmoothL1Loss()

        for p in self.target_model.parameters():
            p.requires_grad = False

        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.update_target_every = update_target_every

        self.n_steps = 0
    
    def _calc_state_size(self):
        # observation contains positions (num_agents,2) and doses (num_agents,)
        positions_dim = self.num_agents * 2
        doses_dim = self.num_agents
        return positions_dim + doses_dim

    def _flatten_state(self, state):
        '''
        Flattens observation dict into a 1D float vector.
        Positions are normalized by grid size, doses are assumed already normalized [0,1].
        '''
        positions = np.asarray(state['positions'], dtype=np.float32) / max(self.env.size - 1, 1)
        doses = np.asarray(state['doses'], dtype=np.float32)
        flat = np.concatenate([positions.flatten(), doses])
        return flat

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def epsilon_greedy_policy(self, state):
        flat_state = self._flatten_state(state)
        state_tensor = torch.as_tensor(flat_state, dtype=torch.float32)
        q_values = self.model(state_tensor.unsqueeze(0)).squeeze(0)  # shape (num_agents, 4)

        actions = np.zeros(self.num_agents, dtype=np.int64)
        for agent_idx in range(self.num_agents):
            if np.random.rand() < self.epsilon:
                actions[agent_idx] = np.random.randint(0, 4)
            else:
                actions[agent_idx] = torch.argmax(q_values[agent_idx]).item()
        return actions

    def compute_target_values(self, next_states, rewards, dones):
        # next_states: tensor (batch, state_size)
        # rewards: tensor (batch, num_agents)
        # dones: tensor (batch,)
        with torch.no_grad():
            q_next = self.target_model(next_states)  # (batch, num_agents, 4)
            max_next = torch.max(q_next, dim=-1).values  # (batch, num_agents)
            dones_expanded = dones.unsqueeze(1).expand_as(max_next)
            targets = rewards + (1 - dones_expanded) * self.gamma * max_next
        return targets
    
    def replay(self):
        if len(self.replay_buffer) > self.batch_size:
            minibatch = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*minibatch))

            states = torch.as_tensor(np.stack(states), dtype=torch.float32)
            actions = torch.as_tensor(np.stack(actions), dtype=torch.int64)  # (batch, num_agents)
            rewards = torch.as_tensor(np.stack(rewards), dtype=torch.float32)  # (batch, num_agents)
            next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32)
            dones = torch.as_tensor(np.stack(dones), dtype=torch.float32)  # (batch,)

            current_q = self.model(states)  # (batch, num_agents, 4)
            current_q_selected = torch.gather(current_q, dim=2, index=actions.unsqueeze(-1)).squeeze(-1)  # (batch, num_agents)

            target_q = self.compute_target_values(next_states, rewards, dones)

            loss_q = self.loss_fn(current_q_selected, target_q)

            self.optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
            self.optimizer.step()
        
    def memorize(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_episode(self):
        obs, _ = self.env.reset()
        state = self._flatten_state(obs)
        self.reward = 0
        done = False
        for _ in range(self.max_steps):
            actions = self.epsilon_greedy_policy(obs)
            probs, next_obs, rewards, done, _ = self.env.step(actions)
            flat_next_state = self._flatten_state(next_obs)

            self.reward += float(self.reward_aggregator(np.array(rewards)))
            self.memorize(state, actions, rewards, flat_next_state, done)
            self.replay()

            self.n_steps += 1
            if self.n_steps == self.update_target_every:
                self.update_target_model()
                self.n_steps = 0
            obs = next_obs
            state = flat_next_state
            if done: break
        # Apply terminal penalty if episode ended without all agents at target
        if not done and not np.all(self.env._reached_target):
            penalty_vec = np.where(self.env._reached_target, 0.0, -self.env._final_miss_penalty)
            self.reward += float(self.reward_aggregator(penalty_vec))
            # Add a terminal transition to propagate the penalty
            self.memorize(state, np.zeros(self.num_agents, dtype=np.int64), penalty_vec, state, True)
            self.replay()
    
    def create_greedy_policy(self):
        def policy_fn(state):
            flat_state = self._flatten_state(state)
            q_values = self.model(torch.as_tensor(flat_state, dtype=torch.float32).unsqueeze(0)).squeeze(0)
            actions = torch.argmax(q_values, dim=-1).detach().cpu().numpy()
            return actions
        return policy_fn
