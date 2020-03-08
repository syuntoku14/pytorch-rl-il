import torch
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from rlil.environments import action_decorator, Action
from ._agent import Agent

# TODO: noarmalize action

class DDPG(Agent):
    """
    Deep Deterministic Policy Gradient (DDPG).
    DDPG extends the ideas of DQN to a continuous action setting.
    Unlike DQN, which uses a single joint Q/policy network, DDPG uses
    separate networks for approximating the Q-function and approximating the policy.
    The policy network outputs a vector action in some continuous space.
    A small amount of noise is added to aid exploration. The Q-network
    is used to train the policy network. A replay buffer is used to
    allow for batch updates and decorrelation of the samples.
    https://arxiv.org/abs/1509.02971

    Args:
        q (QContinuous): An Approximation of the continuous action Q-function.
        policy (DeterministicPolicy): An Approximation of a deterministic policy.
        replay_buffer (ReplayBuffer): The experience replay buffer.
        action_space (gym.spaces.Box): Description of the action space.
        discount_factor (float): Discount factor for future rewards.
        minibatch_size (int): The number of experiences to sample in each training update.
        noise (float): the amount of noise to add to each action (before scaling).
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of timesteps per training update.
    """
    def __init__(self,
                 q,
                 policy,
                 replay_buffer,
                 action_space,
                 discount_factor=0.99,
                 minibatch_size=32,
                 noise=0.1,
                 replay_start_size=5000,
                 update_frequency=1,
                 device=torch.device("cpu")
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.replay_buffer = replay_buffer
        self.device = device
        # hyperparameters
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        self._noise = Normal(0, noise * torch.tensor((action_space.high - action_space.low) / 2).to(self.device))
        self._low = torch.tensor(action_space.low, device=self.device)
        self._high = torch.tensor(action_space.high, device=self.device)
        self._states = None
        self._actions = None
        self._train_count = 0

    def act(self, states, reward):
        self.replay_buffer.store(self._states, self._actions, reward, states)
        self._train()
        self._states = states
        self._actions = self._choose_actions(states)
        return self._actions

    @action_decorator
    def _choose_actions(self, states):
        actions = self.policy.eval(states.to(self.device))
        actions += self._noise.sample([actions.shape[0]])
        actions = torch.min(actions, self._high)
        actions = torch.max(actions, self._low)
        return actions.to("cpu")

    def _train(self):
        if self._should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(self.minibatch_size, device=self.device)

            # train q-network
            q_values = self.q(states, actions.features)
            targets = rewards + self.discount_factor * self.q.target(next_states, self.policy.target(next_states))
            loss = mse_loss(q_values, targets)
            self.q.reinforce(loss)

            # train policy
            greedy_actions = self.policy(states)
            loss = -self.q(states, greedy_actions).mean()
            self.policy.reinforce(loss)
            self.q.zero_grad()

    def _should_train(self):
        self._train_count += 1
        return len(self.replay_buffer) > self.replay_start_size and self._train_count % self.update_frequency == 0