from abc import ABC, abstractmethod
import numpy as np
import torch
from rlil.environments import State, Action
from rlil.initializer import get_device, is_debug_mode
from .replay_buffer import ExperienceReplayBuffer
from .base import BaseBufferWrapper


class GaeWrapper(BaseBufferWrapper):
    """
    A wrapper of ExperienceReplayBuffer for Generalized Advantage Estimation.
    https://arxiv.org/abs/1506.02438
    """

    def __init__(self, buffer, discount_factor=1, lam=1):
        """
        Args:
            buffer (rlil.memory.ExperienceReplayBuffer): 
                A replay_buffer for sampling.
        """
        self.buffer = buffer
        self.device = get_device()
        self.discount_factor = discount_factor
        self.lam = lam

    def compute_gae(self, rewards, values, next_values, masks):
        td_errors = rewards + self.discount_factor * next_values - values

        # compute_gaes
        length = len(td_errors)
        gaes = torch.zeros(length, device=self.device)

        gae = 0.0
        for i in reversed(range(length)):
            mask = masks[i].float()
            gae = td_errors[i] + self.discount_factor * self.lam * gae * mask
            gaes[i] = gae

        # normalize Advantage
        # see: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/issues/102
        gaes = (gaes - gaes.mean()) / gaes.std()
        return gaes