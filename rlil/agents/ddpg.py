from copy import deepcopy
import torch
from torch.distributions.normal import Normal
from torch.nn.functional import mse_loss
from rlil.environments import Action
from rlil.initializer import get_device, get_replay_buffer
from rlil.memory import ExperienceReplayBuffer
from .base import Agent, LazyAgent


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
        discount_factor (float): Discount factor for future rewards.
        minibatch_size (int): The number of experiences to sample in each training update.
        noise (float): the amount of noise to add to each action (before scaling).
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        update_frequency (int): Number of timesteps per training update.
    """

    def __init__(self,
                 q,
                 policy,
                 discount_factor=0.99,
                 minibatch_size=32,
                 noise=0.1,
                 replay_start_size=5000,
                 update_frequency=1,
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.replay_buffer = get_replay_buffer()
        self.device = get_device()
        # hyperparameters
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        action_space = Action.action_space()
        self._noise = Normal(
            0, noise * torch.tensor((action_space.high - action_space.low) / 2).to(self.device))
        self._states = None
        self._actions = None
        self._train_count = 0

    def act(self, states, reward=None):
        if reward is not None:
            self.replay_buffer.store(
                self._states, self._actions, reward, states)
        self._states = states
        actions = self.policy.eval(states.to(self.device))
        actions += self._noise.sample([actions.shape[0]])
        self._actions = Action(actions).to("cpu")
        return self._actions

    def train(self):
        if self._should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states, _) = self.replay_buffer.sample(
                self.minibatch_size)

            # train q-network
            q_values = self.q(states, actions)
            targets = rewards + self.discount_factor * \
                self.q.target(next_states, Action(
                    self.policy.target(next_states)))
            loss = mse_loss(q_values, targets)
            self.q.reinforce(loss)

            # train policy
            greedy_actions = Action(self.policy(states))
            loss = -self.q(states, greedy_actions).mean()
            self.policy.reinforce(loss)
            self.policy.zero_grad()
            self.q.zero_grad()

    def _should_train(self):
        self._train_count += 1
        return len(self.replay_buffer) > self.replay_start_size and self._train_count % self.update_frequency == 0

    def make_lazy_agent(self):
        model = deepcopy(self.policy.model)
        noise = Normal(0, self._noise.stddev.to("cpu"))
        return DDPGLazyAgent(model.to("cpu"), noise)


class DDPGLazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """
    def __init__(self, policy_model, noise):
        self._replay_buffer = ExperienceReplayBuffer(1e9)
        self._policy_model = policy_model
        self._noise = noise
        self._states = None
        self._actions = None

    def act(self, states, reward):
        self._replay_buffer.store(
            self._states, self._actions, reward, states)
        self._states = states
        with torch.no_grad():
            actions = self._policy_model(states)
            actions += self._noise.sample([actions.shape[0]])
        self._actions = Action(actions)
        return self._actions
