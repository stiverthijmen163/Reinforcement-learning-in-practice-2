"""PPO agent implementation.

The agent is designed for a continuous observation space and a discrete action space.
For our robot task, the policy outputs a distribution over the movement actions.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from agents.base_agent import BaseAgent


class PPOModel(nn.Module):
    """Actor-critic neural network used by PPO.
    The first part of the network is shared between actor and critic.
    The actor head outputs action logits.
    The critic head outputs the state-value estimate V(s).
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        
        # Neural network architecture as described in the paper: "Proximal Policy Optimization Algorithms"
        # Tanh activations keep hidden activations bounded, which can help PPO stability.
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Actor: produces one logit for each discrete action
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic: estimates V(s), the expected return from the current state
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """Return action logits and value estimate for a batch of states"""
        x = self.net(x)

        action_logits = self.actor(x)
        state_value = self.critic(x).squeeze(-1)

        return action_logits, state_value


class PPOAgent(BaseAgent):
    """PPO agent with rollout-based learning

    First stores a rollout of transitions,
    then performs several minibatch updates using the PPO clipped objective
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        update_epochs=4,
        batch_size=64,
        value_coef=0.5,
        entropy_coef=0.01,
        hidden_dim=64,
        device=None,
    ):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # PPO hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        
        # Optimization settings
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.model = PPOModel(state_dim, action_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Rollout memory is cleared after each PPO update
        self.clear_memory()

    def clear_memory(self):
        """Clear the rollout buffer"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def _to_tensor(self, state):
        """Convert one observation to a PyTorch tensor"""
        state = np.array(state, dtype=np.float32)
        state = torch.tensor(state, dtype=torch.float32, device=self.device)
        return state.unsqueeze(0)

    def choose_action(self, state):
        """Sample an action from the current policy"""
        state_tensor = self._to_tensor(state)

        with torch.no_grad():
            logits, value = self.model(state_tensor)
            dist = Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def choose_greedy_action(self, state):
        """Choose the action with the highest policy probability"""
        state_tensor = self._to_tensor(state)

        with torch.no_grad():
            logits, _ = self.model(state_tensor)
            action = torch.argmax(logits, dim=-1)

        return action.item()

    def take_action(self, state):
        # need to have this as we extend BaseAgent
        return self.choose_greedy_action(state)

    def update(self, state, reward, action):
        # need to have this as we extend BaseAgent
        pass

    def remember(self, state, action, log_prob, reward, done, value):
        """Store one transition in the rollout buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def get_value(self, state):
        """Return the critic's estimate V(s) for bootstrapping"""
        state_tensor = self._to_tensor(state)

        with torch.no_grad():
            _, value = self.model(state_tensor)

        return value.item()

    def compute_advantages(self, last_value):
        """Compute GAE advantages and return targets"""
        advantages = []
        gae = 0

        values = self.values + [last_value]

        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                next_non_terminal = 0
            else:
                next_non_terminal = 1

            # single step TD error
            delta = (
                self.rewards[t]
                + self.gamma * values[t + 1] * next_non_terminal
                - values[t]
            )

            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)

        returns = np.array(advantages) + np.array(self.values)

        return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)

    def learn(self, last_value=0):
        """Update the PPO policy using the collected rollout"""
        if len(self.states) == 0:
            return

        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32, device=self.device)

        advantages, returns = self.compute_advantages(last_value)

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        n_samples = len(states)

        for _ in range(self.update_epochs):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                logits, values = self.model(batch_states)
                dist = Categorical(logits=logits)

                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)

                unclipped_objective = ratio * batch_advantages
                clipped_ratio = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon,
                )
                clipped_objective = clipped_ratio * batch_advantages

                actor_loss = -torch.min(
                    unclipped_objective,
                    clipped_objective,
                ).mean()

                critic_loss = nn.functional.mse_loss(values, batch_returns)

                # Subtract entropy to encourage exploration
                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.clear_memory()

    def save(self, path, obs_mode: str = "both") -> None:
        """Save ppo agent and obs_mode to a .pt file"""
        torch.save({
            "agent_type": "ppo",
            "obs_mode":   obs_mode,
            "state_dict": self.model.state_dict(),
        }, path)

    @classmethod
    def load(cls, path) -> 'PPOAgent':
        """Load saved PPOAgent"""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(path, weights_only=False, map_location=device)
        state_dict = checkpoint["state_dict"]
        agent = cls(
            state_dim  = state_dict["net.0.weight"].shape[1],
            action_dim = state_dict["actor.weight"].shape[0],
            hidden_dim = state_dict["net.0.weight"].shape[0],
        )
        agent.model.load_state_dict(state_dict)
        agent.obs_mode = checkpoint["obs_mode"]
        return agent