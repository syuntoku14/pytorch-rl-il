from abc import ABC, abstractmethod
import numpy as np
import torch
from rlil.environments import State, Action
from rlil.initializer import get_device, is_debug_mode
from .replay_buffer import ExperienceReplayBuffer
from .base import BaseBufferWrapper
from .gae_wrapper import GaeWrapper


class GailWrapper(BaseBufferWrapper):
    """
    A wrapper of ExperienceReplayBuffer for rlil.agents.GAIL.
    """

    def __init__(self, buffer, expert_buffer, discriminator):
        """
        Args:
            buffer (rlil.memory.ExperienceReplayBuffer): 
                A replay_buffer for sampling.
            expert_buffer (rlil.memory.ExperienceReplayBuffer):
                A replay_buffer with expert trajectories.
            discriminator (rlil.approximation.Discriminator):
                A discriminator approximation.
        """
        self.buffer = buffer
        self.expert_buffer = expert_buffer
        self.device = get_device()
        self.discriminator = discriminator

    def sample(self, batch_size):
        # replace the rewards with gail rewards
        states, actions, rewards, next_states, weights, indexes = \
            self.buffer.sample(batch_size)

        rewards = self.discriminator.expert_reward(
            torch.cat((states.features, actions.features), dim=1))
        return (states, actions, rewards, next_states, weights, indexes)

    def sample_both(self, batch_size):
        batch_size = int(batch_size / 2)
        samples = self.buffer.sample(batch_size)
        expert_samples = self.expert_buffer.sample(batch_size)
        return samples, expert_samples

    def get_all_transitions(self):
        # return the sampled trajectories
        # not including expert trajectories
        return self.buffer.get_all_transitions()

    def compute_gae(self, *args, **kwargs):
        # wrap function for GaeWrapper
        if isinstance(self.buffer, GaeWrapper):
            return self.buffer.compute_gae(*args, **kwargs)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        # return the number of sampled trajectories
        # not including expert trajectories
        return len(self.buffer)
