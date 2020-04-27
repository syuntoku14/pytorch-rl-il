import torch
from torch.nn.functional import mse_loss
from .base import Agent, LazyAgent
from rlil.initializer import (get_replay_buffer,
                              get_device,
                              get_replay_buffer)


class PPO(Agent):
    """
    Proximal Policy Optimization (PPO).
    PPO is an actor-critic style policy gradient algorithm that allows for the reuse of samples
    by using importance weighting. This often increases sample efficiency compared to algorithms
    such as A2C. To avoid overfitting, PPO uses a special "clipped" objective that prevents
    the algorithm from changing the current policy too quickly.

    Args:
        features (FeatureNetwork): Shared feature layers.
        v (VNetwork): Value head which approximates the state-value function.
        policy (StochasticPolicy): Policy head which outputs an action distribution.
        discount_factor (float): Discount factor for future rewards.
        entropy_loss_scaling (float): Contribution of the entropy loss to the total policy loss.
        epochs (int): Number of times to reuse each sample.
        lam (float): The Generalized Advantage Estimate (GAE) decay parameter.
        minibatches (int): The number of minibatches to split each batch into.
        writer (Writer): Used for logging.
    """

    def __init__(
            self,
            features,
            v,
            policy,
            discount_factor=0.99,
            entropy_loss_scaling=0.01,
            epochs=4,
            epsilon=0.2,
            lam=0.95,
            minibatches=4,
    ):
        # objects
        self.features = features
        self.v = v
        self.policy = policy
        self.replay_buffer = get_replay_buffer()
        self.writer = get_writer()
        self.device = get_device()
        # hyperparameters
        self.discount_factor = discount_factor
        self.entropy_loss_scaling = entropy_loss_scaling
        self.epochs = epochs
        self.epsilon = epsilon
        self.lam = lam
        self.minibatches = minibatches
        # private
        self._states = None
        self._actions = None

    def act(self, states, rewards=None):
        if reward is not None:
            self.replay_buffer.store(
                self._states, self._actions, reward, states)
        self._states = states
        self._actions = Action(self.policy.no_grad(
            self.features(states.to(self.device))).sample())
        return self._actions

    def train(self):
        # load trajectories from buffer
        (states, actions, rewards, next_states, _) = self.replay_buffer.sample(
            self.minibatch_size)

        # compute target values
        features = self.features.no_grad(states)
        pi_0 = self.policy.no_grad(features).log_prob(actions)
        targets = self.v.no_grad(features) + advantages

        # train for several epochs
        for _ in range(self.epochs):
            self._train_epoch(states, actions, advantages, targets, pi_0)

    def _train_epoch(self, states, actions, advantages, targets, pi_0):
        # randomly permute the indexes to generate minibatches
        minibatch_size = int(self._batch_size / self.minibatches)
        indexes = torch.randperm(self._batch_size)
        for n in range(self.minibatches):
            # load the indexes for the minibatch
            first = n * minibatch_size
            last = first + minibatch_size
            i = indexes[first:last]

            # perform a single training step
            self._train_minibatch(
                states[i], actions[i], pi_0[i], advantages[i], targets[i])

    def _train_minibatch(self, states, actions, pi_0, advantages, targets):
        # forward pass
        features = self.features(states)
        values = self.v(features)
        distribution = self.policy(features)
        pi_i = distribution.log_prob(actions)

        # compute losses
        value_loss = mse_loss(values, targets)
        policy_gradient_loss = self._clipped_policy_gradient_loss(
            pi_0, pi_i, advantages)
        entropy_loss = -distribution.entropy().mean()
        policy_loss = policy_gradient_loss + self.entropy_loss_scaling * entropy_loss

        # backward pass
        self.v.reinforce(value_loss)
        self.policy.reinforce(policy_loss)
        self.features.reinforce()

        # debugging
        self.writer.add_loss('policy_gradient', policy_gradient_loss.detach())
        self.writer.add_loss('entropy', entropy_loss.detach())

    def _clipped_policy_gradient_loss(self, pi_0, pi_i, advantages):
        ratios = torch.exp(pi_i - pi_0)
        surr1 = ratios * advantages
        epsilon = self.epsilon
        surr2 = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages
        return -torch.min(surr1, surr2).mean()
