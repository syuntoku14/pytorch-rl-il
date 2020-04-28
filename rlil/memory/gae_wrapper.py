from abc import ABC, abstractmethod
import numpy as np
import torch
from rlil.environments import State, Action
from rlil.utils.optim import Schedulable
from rlil.initializer import get_device, is_debug_mode
from .replay_buffer import ExperienceReplayBuffer


class GaeWrapper(ExperienceReplayBuffer):
    """
    A wrapper of ExperienceReplayBuffer for Generalized Advantage Estimation.
    https://arxiv.org/abs/1506.02438
    """

    def __init__(self, buffer, gamma=1, lam=1):
        """
        Args:
            buffer (rlil.memory.ExperienceReplayBuffer): 
                A replay_buffer for sampling.
        """
        self.buffer = buffer
        self.device = get_device()
        self.gamma = gamma
        self.lam = lam

    def store(self, *args, **kwargs):
        self.buffer.store(*args, **kwargs)

    def sample(self, *args, **kwargs):
        pass

    def get_all_transitions(self):
        return self.buffer.get_all_transitions()

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)

    def compute_gae(self, states, rewards, next_states, features, v):
        values = v.target(features.target(states))
        next_values = v.target(features.target(next_states))
        td_errors = rewards + self.gamma * next_values - values

        # compute_gaes
        gaes = td_errors.clone().view(-1)
        length = len(td_errors)

        gae = 0
        for i in reversed(range(length)):
            mask = next_states[i].mask.float()
            gae = td_errors[i] + self.gamma * self.lam * gae * mask
            gaes[i] = gae

        return gaes
