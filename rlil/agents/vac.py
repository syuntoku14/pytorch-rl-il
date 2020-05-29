import torch
from torch.nn.functional import mse_loss
from copy import deepcopy
from rlil.environments import Action
from rlil.initializer import (get_replay_buffer,
                              get_device,
                              get_writer)
from .base import Agent, LazyAgent
from rlil.utils import Samples


class VAC(Agent):
    '''
    Vanilla Actor-Critic (VAC).
    VAC is an implementation of the actor-critic algorithm found in the Sutton and Barto (2018) textbook.
    This implementation tweaks the algorithm slightly by using a shared feature layer.
    In addition to that, this implementation uses batched states and actions to update parameters.
    https://papers.nips.cc/paper/1786-actor-critic-algorithms.pdf

    Args:
        feature_nw (FeatureNetwork): Shared feature layers.
        v (VNetwork): Value head which approximates the state-value function.
        policy (StochasticPolicy): Policy head which outputs an action distribution.
        discount_factor (float): Discount factor for future rewards.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
    '''

    def __init__(self,
                 feature_nw,
                 v,
                 policy,
                 discount_factor=1,
                 replay_start_size=500):
        self.feature_nw = feature_nw
        self.v = v
        self.policy = policy
        self.discount_factor = discount_factor
        self.replay_start_size = replay_start_size
        self.replay_buffer = get_replay_buffer()
        self.writer = get_writer()
        self.device = get_device()
        # private
        self._states = None
        self._actions = None

    def act(self, states, rewards=None):
        if rewards is not None:
            samples = Samples(self._states, self._actions, rewards, states)
            self.replay_buffer.store(samples)
        self._states = states
        self._actions = Action(self.policy.no_grad(
            self.feature_nw.no_grad(states.to(self.device))).sample()).to("cpu")
        return self._actions

    def train(self):
        if self.should_train():
            states, actions, rewards, next_states, _, _ = \
                self.replay_buffer.get_all_transitions()

            # forward pass
            features = self.feature_nw(states)
            values = self.v(features)
            log_prob = self.policy(features).log_prob(actions.raw)

            # compute targets
            targets = rewards + self.discount_factor * \
                self.v.target(self.feature_nw.target(next_states))

            # compute losses
            value_loss = mse_loss(values, targets)
            advantages = targets - values.detach()
            policy_loss = - (advantages * log_prob).mean()

            # backward pass
            self.policy.reinforce(policy_loss)
            self.v.reinforce(value_loss)
            self.feature_nw.reinforce()
            self.writer.train_steps += 1

            # clear buffer for on-policy training
            self.replay_buffer.clear()

    def should_train(self):
        return len(self.replay_buffer) > self.replay_start_size

    def make_lazy_agent(self, evaluation=False, store_samples=True):
        policy_model = deepcopy(self.policy.model)
        feature_model = deepcopy(self.feature_nw.model)
        return VacLazyAgent(policy_model.to("cpu"),
                            feature_model.to("cpu"),
                            evaluation=evaluation,
                            store_samples=store_samples)


class VacLazyAgent(LazyAgent):
    """ 
    Agent class for sampler.
    """

    def __init__(self, policy_model, feature_model, *args, **kwargs):
        self._feature_model = feature_model
        self._policy_model = policy_model
        super().__init__(*args, **kwargs)
        if self._evaluation:
            self._feature_model.eval()
            self._policy_model.eval()

    def act(self, states, reward):
        super().act(states, reward)
        self._states = states
        with torch.no_grad():
            if self._evaluation:
                outputs = self._policy_model(self._feature_model(states),
                                             return_mean=True)
            else:
                outputs = self._policy_model(
                    self._feature_model(states)).sample()
            self._actions = Action(outputs).to("cpu")
        return self._actions
