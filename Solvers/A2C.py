import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.optim import Adam

from Solvers.AbstractSolver import AbstractSolver


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_sizes=[64, 64]):
        super().__init__()
        hidden_sizes = hidden_sizes or [64]
        layers = []
        input_size = state_size
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden))
            input_size = hidden
        self.layers = nn.ModuleList(layers)
        self.policy_head = nn.Linear(input_size, action_size)
        self.value_head = nn.Linear(input_size, 1)

    def forward(self, obs):
        if not torch.is_tensor(obs):
            x = torch.as_tensor(obs, dtype=torch.float32)
        else:
            x = obs.float()

        if x.ndim == 1:
            x = x.unsqueeze(0)

        for layer in self.layers:
            x = torch.relu(layer(x))

        probs = torch.softmax(self.policy_head(x), dim=-1)
        value = self.value_head(x).squeeze(-1)

        return probs.squeeze(0), value.squeeze(0)


class A2C(AbstractSolver):
    def __init__(self, env, epsilon=0.1, gamma=0.1, num_episodes=100, max_steps=100, layers=[64, 64], lr=1e-3):
        super().__init__(env, epsilon, gamma, num_episodes, max_steps)
        self.actor_critic = ActorCriticNetwork(
            env.observation_space.n, env.action_space.n, layers
        )
        self.optimizer = Adam(self.actor_critic.parameters(), lr=lr)

    def _encode_state(self, state):
        state_tensor = torch.as_tensor(state)
        if state_tensor.ndim == 0:
            state_tensor = state_tensor.unsqueeze(0)
        state_tensor = state_tensor.long()
        encoded = F.one_hot(state_tensor, num_classes=self.env.observation_space.n).float()
        if encoded.shape[0] == 1:
            return encoded.squeeze(0)
        return encoded

    def create_greedy_policy(self):
        def policy_fn(state):
            encoded_state = self._encode_state(state)
            return torch.argmax(self.actor_critic(encoded_state)[0]).detach().item()

        return policy_fn

    def select_action(self, state):
        encoded_state = self._encode_state(state)
        probs, value = self.actor_critic(encoded_state)

        probs_np = probs.detach().numpy()
        action = np.random.choice(len(probs_np), p=probs_np)

        return action, probs[action], value

    def actor_loss(self, advantage, prob):
        return -advantage * torch.log(prob)

    def critic_loss(self, advantage):
        return 0.5 * advantage.pow(2)

    def update_actor_critic(self, advantage, prob):
        actor_loss = self.actor_loss(advantage.detach(), prob)
        critic_loss = self.critic_loss(advantage)

        loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train_episode(self):
        state, _ = self.env.reset()
        self.reward = 0.0

        for _ in range(self.max_steps):
            action, prob, value = self.select_action(state)
            next_state, reward, done, _ = self.step(action)
            self.reward += reward
            encoded_next_state = self._encode_state(next_state)
            _, next_value = self.actor_critic(encoded_next_state)

            advantage = reward + (1 - done) * self.gamma * next_value - value
            self.update_actor_critic(advantage, prob)

            state = next_state
            if done:
                break
