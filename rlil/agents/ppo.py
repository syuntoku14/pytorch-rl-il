import torch
import os
from torch.nn.functional import mse_loss
from .base import Agent, LazyAgent
from copy import deepcopy
from rlil.environments import Action
from rlil.initializer import (get_replay_buffer,
                              get_device,
                              get_writer)
from rlil.utils import Samples


class PPO(Agent):
    """
    Proximal Policy Optimization (PPO).
    PPO is an actor-critic style policy gradient algorithm that allows for the reuse of samples
    by using importance weighting. This often increases sample efficiency compared to algorithms
    such as A2C. To avoid overfitting, PPO uses a special "clipped" objective that prevents
    the algorithm from changing the current policy too quickly.

    Args:
        feature_nw (FeatureNetwork): Shared feature layers.
        v (VNetwork): Value head which approximates the state-value function.
        policy (StochasticPolicy): Policy head which outputs an action distribution.
        entropy_loss_scaling (float): Contribution of the entropy loss to the total policy loss.
        epochs (int): Number of times to reuse each sample.
        lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
        replay_start_size (int): Number of experiences in replay buffer when training begins.
        minibatches (int): The number of minibatches to split each batch into.
        epochs (int): Number of times to reuse each sample.
    """

    def __init__(
            self,
            feature_nw,
            v,
            policy,
            entropy_loss_scaling=0.01,
            epsilon=0.2,
            replay_start_size=5000,
            minibatches=4,
            epochs=4,
    ):
        # objects
        self.feature_nw = feature_nw
        self.v = v
        self.policy = policy
        self.replay_buffer = get_replay_buffer()
        self.writer = get_writer()
        self.device = get_device()
        # hyperparameters
        self.entropy_loss_scaling = entropy_loss_scaling
        self.epsilon = epsilon
        self.minibatches = minibatches
        self.epochs = epochs
        self.replay_start_size = replay_start_size
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

            # compute gae
            features = self.feature_nw.target(states)
            values = self.v.target(features)
            next_values = self.v.target(self.feature_nw.target(next_states))
            advantages = self.replay_buffer.compute_gae(
                rewards, values, next_values, next_states.mask)

            # compute target values
            # actions.raw is used since .features clip the actions
            pi_0 = self.policy.no_grad(features).log_prob(actions.raw)
            targets = values + advantages

            # train for several epochs
            for _ in range(self.epochs):
                # randomly permute the indexes to generate minibatches
                minibatch_size = int(len(states) / self.minibatches)
                indexes = torch.randperm(len(states))
                for n in range(self.minibatches):
                    # load the indexes for the minibatch
                    first = n * minibatch_size
                    last = first + minibatch_size
                    i = indexes[first:last]

                    # perform a single training step
                    self._train_minibatch(
                        states[i], actions[i], pi_0[i], advantages[i], targets[i])
                    self.writer.train_steps += 1

            # clear buffer for on-policy training
            self.replay_buffer.clear()

    def _train_minibatch(self, states, actions, pi_0, advantages, targets):
        # forward pass
        features = self.feature_nw(states)
        values = self.v(features)
        distribution = self.policy(features)
        pi_i = distribution.log_prob(actions.raw)

        # compute losses
        value_loss = mse_loss(values, targets).mean()
        policy_gradient_loss = self._clipped_policy_gradient_loss(
            pi_0, pi_i, advantages)
        entropy_loss = -distribution.entropy().mean()
        policy_loss = policy_gradient_loss + \
            self.entropy_loss_scaling * entropy_loss

        # backward pass
        self.policy.reinforce(policy_loss)
        self.v.reinforce(value_loss)
        self.feature_nw.reinforce()

        # debugging
        self.writer.add_scalar('loss/policy_gradient',
                               policy_gradient_loss.detach())
        self.writer.add_scalar('loss/entropy',
                               entropy_loss.detach())

    def _clipped_policy_gradient_loss(self, pi_0, pi_i, advantages):
        ratios = torch.exp(pi_i - pi_0)
        # debugging
        self.writer.add_scalar('loss/ratios/max', ratios.max())
        self.writer.add_scalar('loss/ratios/min', ratios.min())
        surr1 = ratios * advantages
        epsilon = self.epsilon
        surr2 = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages
        return -torch.min(surr1, surr2).mean()

    def should_train(self):
        return len(self.replay_buffer) > self.replay_start_size

    def make_lazy_agent(self, evaluation=False, store_samples=True):
        policy_model = deepcopy(self.policy.model)
        feature_model = deepcopy(self.feature_nw.model)
        return PPOLazyAgent(policy_model.to("cpu"),
                            feature_model.to("cpu"),
                            evaluation=evaluation,
                            store_samples=store_samples)

    def load(self, dirname):
        for filename in os.listdir(dirname):
            if filename == 'policy.pt':
                self.policy.model = torch.load(os.path.join(
                    dirname, filename), map_location=self.device)
            if filename in ('feature.pt'):
                self.feature_nw.model = torch.load(os.path.join(dirname, filename),
                                                   map_location=self.device)
            if filename in ('v.pt'):
                self.v.model = torch.load(os.path.join(dirname, filename),
                                          map_location=self.device)


class PPOLazyAgent(LazyAgent):
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
