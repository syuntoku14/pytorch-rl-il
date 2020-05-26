from abc import ABC, abstractmethod
import numpy as np
import torch
from rlil.environments import State, Action
from rlil.initializer import get_device, is_debug_mode
from .gail_wrapper import GailWrapper


class AirlWrapper(GailWrapper):
    """
    A wrapper of ExperienceReplayBuffer for rlil.agents.AIRL.
    """

    def __init__(self,
                 buffer,
                 expert_buffer,
                 reward_fn,
                 value_fn,
                 policy,
                 feature_nw=None,
                 discount_factor=1.0):
        """
        Args:
            buffer (rlil.memory.ExperienceReplayBuffer): 
                A replay_buffer for sampling.
            expert_buffer (rlil.memory.ExperienceReplayBuffer):
                A replay_buffer with expert trajectories.
            reward_fn (rlil.approximation.Approximation):
                A reward function approximation.
            value_fn (rlil.approximation.Approximation):
                A value function approximation.
            policy (rlil.policies):
                A policy approximation
            feature_nw (rlil.approximation.FeatureNetwork)
        """
        self.buffer = buffer
        self.expert_buffer = expert_buffer
        self.device = get_device()
        self.reward_fn = reward_fn
        self.value_fn = value_fn
        self.policy = policy
        self.feature_nw = feature_nw
        self.discount_factor = discount_factor

    def sample(self, batch_size):
        # replace the rewards with gail rewards
        states, actions, rewards, next_states, weights, indexes = \
            self.buffer.sample(batch_size)

        ds = self.discrim(states, actions, next_states)
        rewards = self.expert_reward(ds)
        return (states, actions, rewards, next_states, weights, indexes)

    def discrim(self, states, actions, next_states):
        if self.feature_nw is None:
            features = states
        else:
            features = self.feature_nw.no_grad(states)
        policy_prob = self.policy.no_grad(features).log_prob(
            actions.features).exp()

        f = self.reward_fn(
            torch.cat((states.features, actions.features), dim=1)).squeeze(1) \
            + next_states.mask.float() \
            * (self.discount_factor * self.value_fn(next_states)
               - self.value_fn(states))
        f_exp = f.exp()
        d = f_exp / (f_exp + policy_prob)
        return d

    def expert_reward(self, d):
        return (torch.log(d) - torch.log(1 - d)).squeeze().detach()
