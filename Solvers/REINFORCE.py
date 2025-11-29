import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam
from Solvers.AbstractSolver import AbstractSolver


class PolicyNet(nn.Module):
    def __init__(self, state_size, num_agents, hidden_sizes=[128, 128]):
        super().__init__()
        hidden_sizes = hidden_sizes or [128]
        layers = []
        input_size = state_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden))
            input_size = hidden
        self.layers = nn.ModuleList(layers)
        self.policy_head = nn.Linear(input_size, num_agents * 4)
        self.value_head = nn.Linear(input_size, 1)
        self.num_agents = num_agents

    def forward(self, obs):
        if not torch.is_tensor(obs):
            x = torch.as_tensor(obs, dtype=torch.float32)
        else:
            x = obs.float()

        if x.ndim == 1:
            x = x.unsqueeze(0)

        for layer in self.layers:
            x = torch.relu(layer(x))

        logits = self.policy_head(x)  # (batch, num_agents*4)
        logits = logits.view(x.shape[0], self.num_agents, 4)
        probs = torch.softmax(logits, dim=-1)  # (batch, num_agents, 4)
        baseline = self.value_head(x).squeeze(-1)  # (batch,)

        return probs, baseline


class Reinforce(AbstractSolver):
    def __init__(self, env, epsilon=0.1, gamma=0.1, num_episodes=100, max_steps=100, layers=[128, 128], lr=1e-3):
        super().__init__(env, epsilon, gamma, num_episodes, max_steps)
        self.num_agents = env.num_agents
        self.state_size = self._calc_state_size()
        self.model = PolicyNet(self.state_size, self.num_agents, layers)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
    
    def _calc_state_size(self):
        positions_dim = self.num_agents * 2
        doses_dim = self.num_agents
        return positions_dim + doses_dim

    def _flatten_state(self, state):
        positions = np.asarray(state['positions'], dtype=np.float32) / max(self.env.size - 1, 1)
        doses = np.asarray(state['doses'], dtype=np.float32)
        flat = np.concatenate([positions.flatten(), doses])
        return flat
    
    def create_greedy_policy(self):
        def policy_fn(state):
            flat_state = self._flatten_state(state)
            probs, _ = self.model(torch.as_tensor(flat_state, dtype=torch.float32))
            actions = torch.argmax(probs.squeeze(0), dim=-1).detach().cpu().numpy()
            return actions
        return policy_fn

    def compute_returns(self, rewards):
        returns = [0]*len(rewards)
        G = 0
        for i in range(len(returns)-1, -1, -1):
            G = rewards[i] + self.gamma*G
            returns[i] = G
        return returns

    def select_action(self, state):
        flat_state = self._flatten_state(state)
        probs, baseline = self.model(torch.as_tensor(flat_state, dtype=torch.float32))
        probs = probs.squeeze(0)  # (num_agents, 4)

        actions = []
        log_probs = []
        for agent_idx in range(self.num_agents):
            p = probs[agent_idx]
            a = np.random.choice(4, p=p.detach().numpy())
            actions.append(a)
            log_probs.append(torch.log(p[a]))

        actions = np.array(actions, dtype=np.int64)
        log_prob_sum = torch.stack(log_probs).sum()

        return actions, log_prob_sum, baseline.squeeze(0)
    
    def pg_loss(self, advantage, log_prob_sum):
        return -advantage * log_prob_sum
    
    def update_model(self, rewards, log_prob_sums, baselines):
        returns = torch.as_tensor(
            self.compute_returns(rewards), dtype=torch.float32
        )

        log_prob_sums = torch.stack(log_prob_sums)
        baselines = torch.stack(baselines)

        deltas = returns - baselines

        pg_loss = self.pg_loss(deltas.detach(), log_prob_sums).mean()
        value_loss = F.smooth_l1_loss(returns.detach(), baselines)

        loss = pg_loss + value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()
    
    def train_episode(self):
        obs, _ = self.env.reset()
        self.reward = 0.0
        rewards = []
        log_prob_sums = []
        baselines = []
        for _ in range(self.max_steps):
            actions, log_prob_sum, baseline = self.select_action(obs)
            _, next_obs, rewards_vec, done, _ = self.env.step(actions)
            reward_scalar = float(self.reward_aggregator(np.array(rewards_vec)))
            self.reward += reward_scalar
            rewards.append(reward_scalar)
            log_prob_sums.append(log_prob_sum)
            baselines.append(baseline)

            obs = next_obs
            if done: break
        
        self.update_model(rewards, log_prob_sums, baselines)
