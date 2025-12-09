import torch
import torch.nn as nn
import numpy as np

from torch.optim import Adam

from Solvers.AbstractSolver import AbstractSolver


class ActorCriticNetwork(nn.Module):
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
        value = self.value_head(x).squeeze(-1)  # (batch,)

        return probs, value


class A2C(AbstractSolver):
    def __init__(self, env, epsilon=0.1, gamma=0.99, num_episodes=100, max_steps=100, layers=[128, 128], lr=1e-3):
        super().__init__(env, epsilon, gamma, num_episodes, max_steps)
        self.num_agents = env.num_agents
        self.state_size = self._calc_state_size()
        self.actor_critic = ActorCriticNetwork(
            self.state_size, self.num_agents, layers
        )
        self.optimizer = Adam(self.actor_critic.parameters(), lr=lr)

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
            probs, _ = self.actor_critic(torch.as_tensor(flat_state, dtype=torch.float32))
            actions = torch.argmax(probs.squeeze(0), dim=-1).detach().cpu().numpy()
            return actions

        return policy_fn

    def select_action(self, state):
        flat_state = self._flatten_state(state)
        probs, value = self.actor_critic(torch.as_tensor(flat_state, dtype=torch.float32))
        probs = probs.squeeze(0)  # (num_agents, 4)

        actions = []
        action_probs = []
        for agent_idx in range(self.num_agents):
            p = probs[agent_idx]
            a = np.random.choice(4, p=p.detach().numpy())
            actions.append(a)
            action_probs.append(p[a])

        actions = np.array(actions, dtype=np.int64)
        log_prob_sum = torch.stack([torch.log(p) for p in action_probs]).sum()

        return actions, log_prob_sum, value.squeeze(0)

    def actor_loss(self, advantage, log_prob_sum):
        return -advantage * log_prob_sum

    def critic_loss(self, advantage):
        return 0.5 * advantage.pow(2)

    def update_actor_critic(self, advantage, log_prob_sum):
        actor_loss = self.actor_loss(advantage.detach(), log_prob_sum)
        critic_loss = self.critic_loss(advantage)

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor_critic.parameters(), 100)
        self.optimizer.step()

    def train_episode(self):
        obs, _ = self.env.reset()
        self.reward = 0.0
        done = False
        for _ in range(self.max_steps):
            actions, log_prob_sum, value = self.select_action(obs)
            _, next_obs, rewards, done, _ = self.env.step(actions)
            reward_scalar = float(self.reward_aggregator(np.array(rewards)))
            self.reward += reward_scalar

            flat_next_state = self._flatten_state(next_obs)
            _, next_value = self.actor_critic(torch.as_tensor(flat_next_state, dtype=torch.float32))

            advantage = reward_scalar + (1 - done) * self.gamma * next_value.squeeze(0) - value
            self.update_actor_critic(advantage, log_prob_sum)

            obs = next_obs
            if done:
                break
        # If episode ended without all agents at target, apply terminal penalty
        if not done and not np.all(self.env._reached_target):
            penalty_vec = np.where(self.env._reached_target, 0.0, -self.env._final_miss_penalty)
            penalty_scalar = float(self.reward_aggregator(penalty_vec))
            self.reward += penalty_scalar
            flat_state = self._flatten_state(obs)
            _, value = self.actor_critic(torch.as_tensor(flat_state, dtype=torch.float32))
            advantage = penalty_scalar - value
            # No associated action log-prob for this synthetic terminal penalty
            self.update_actor_critic(advantage, torch.tensor(0.0))
