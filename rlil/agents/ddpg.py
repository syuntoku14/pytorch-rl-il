from copy import deepcopy
import torch
import os
from torch.distributions.normal import Normal
from rlil.environments import Action
from rlil.initializer import (
    get_device, get_writer, get_replay_buffer, use_apex)
from rlil.memory import ExperienceReplayBuffer
from rlil.nn import weighted_mse_loss
from rlil.utils import Samples
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
    """

    def __init__(self,
                 q,
                 policy,
                 discount_factor=0.99,
                 minibatch_size=32,
                 noise=0.1,
                 replay_start_size=5000,
                 ):
        # objects
        self.q = q
        self.policy = policy
        self.replay_buffer = get_replay_buffer()
        self.device = get_device()
        self.writer = get_writer()
        # hyperparameters
        self.replay_start_size = replay_start_size
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        # private
        action_space = Action.action_space()
        self._noise = Normal(0,
                             noise*torch.tensor(
                                 (action_space.high - action_space.low) / 2,
                                 dtype=torch.float32, device=self.device))
        self._states = None
        self._actions = None

    def act(self, states, reward=None):
        if reward is not None:
            samples = Samples(self._states, self._actions, reward, states)
            self.replay_buffer.store(samples)
        self._states = states
        actions = self.policy.no_grad(states.to(self.device))
        actions += self._noise.sample([actions.shape[0]])
        self._actions = Action(actions).to("cpu")
        return self._actions

    def train(self):
        if self.should_train():
            # sample transitions from buffer
            (states, actions, rewards, next_states,
             weights, indexes) = self.replay_buffer.sample(self.minibatch_size)

            # train q-network
            q_values = self.q(states, actions)
            targets = rewards + self.discount_factor * \
                self.q.target(next_states, Action(
                    self.policy.target(next_states)))
            q_loss = weighted_mse_loss(q_values, targets, weights)
            self.q.reinforce(q_loss)

            # update prioritized replay buffer
            td_errors = (targets - q_values).abs()
            self.replay_buffer.update_priorities(indexes, td_errors.cpu())

            # train policy
            policy_actions = Action(self.policy(states))
            policy_loss = -self.q(states, policy_actions).mean()
            self.policy.reinforce(policy_loss)

            # additional debugging info
            self.writer.add_scalar('loss/td_error', td_errors.mean())
            self.writer.train_steps += 1

    def should_train(self):
        return len(self.replay_buffer) > self.replay_start_size

    def make_lazy_agent(self, evaluation=False, store_samples=True):
        policy_model = deepcopy(self.policy.model)
        q_model = deepcopy(self.q.model)
        policy_target_model = deepcopy(self.policy._target._target)
        q_target_model = deepcopy(self.q._target._target)
        noise = Normal(0, self._noise.stddev.to("cpu"))
        return DDPGLazyAgent(policy_model=policy_model.to("cpu"),
                             policy_target_model=policy_target_model.to("cpu"),
                             q_model=q_model.to("cpu"),
                             q_target_model=q_target_model.to("cpu"),
                             discount_factor=self.discount_factor,
                             noise=noise,
                             use_apex=use_apex(),
                             evaluation=evaluation,
                             store_samples=store_samples)

    def load(self, dirname):
        for filename in os.listdir(dirname):
            if filename == 'policy.pt':
                self.policy.model = torch.load(os.path.join(
                    dirname, filename), map_location=self.device)
            if filename in ('q.pt'):
                self.q.model = torch.load(os.path.join(dirname, filename),
                                          map_location=self.device)


class DDPGLazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """

    def __init__(self, policy_model, policy_target_model,
                 q_model, q_target_model, discount_factor,
                 noise, use_apex, *args, **kwargs):
        self._policy_model = policy_model
        self._policy_target_model = policy_target_model
        self._q_model = q_model
        self._q_target_model = q_target_model
        self._discount_factor = discount_factor
        self._noise = noise
        self._use_apex = use_apex
        super().__init__(*args, **kwargs)
        if self._evaluation:
            self._policy_model.eval()

    def act(self, states, reward):
        super().act(states, reward)
        self._states = states
        with torch.no_grad():
            actions = self._policy_model(states)
            if not self._evaluation:
                actions += self._noise.sample([actions.shape[0]])
        self._actions = Action(actions)
        return self._actions

    def compute_priorities(self, samples):
        if self._use_apex:
            q_values = self._q_model(samples.states, samples.actions)
            targets = samples.rewards + self._discount_factor * \
                self._q_target_model(samples.next_states, Action(
                    self._policy_target_model(samples.next_states)))

            td_errors = (targets - q_values).abs()
            return td_errors
        else:
            return None
