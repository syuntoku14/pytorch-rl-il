from abc import ABC, abstractmethod
import numpy as np
import torch
from rlil.environments import State, Action
from rlil.utils.optim import Schedulable
from rlil.initializer import get_device, is_debug_mode
from .replay_buffer import ExperienceReplayBuffer
from .base import BaseBufferWrapper
from .gae_wrapper import GaeWrapper


class SqilWrapper(BaseBufferWrapper):
    """
    SQIL is a behavior cloning method which regularizes the 
    reward to sparse by giving the agent a constant 
    reward of r = +1 for matching the demonstrated action in 
    a demonstrated state, and giving the agent a constant reward
    of r = 0 for all other behavior.
    https://arxiv.org/abs/1905.11108
    """

    def __init__(self, buffer, expert_buffer):
        """
        Args:
            buffer (rlil.memory.ExperienceReplayBuffer): 
                A replay_buffer for sampling.
            expert_buffer (rlil.memory.ExperienceReplayBuffer):
                A replay_buffer with expert trajectories.
        """
        self.buffer = buffer
        self.expert_buffer = expert_buffer
        self.device = get_device()

    def sample(self, batch_size):
        batch_size = int(batch_size / 2)
        states, actions, rewards, next_states, weights = \
            self.buffer.sample(batch_size)
        exp_states, exp_actions, exp_rewards, exp_next_states, exp_weights = \
            self.expert_buffer.sample(batch_size)

        rewards = torch.zeros_like(rewards, dtype=torch.float32)
        exp_rewards = torch.ones_like(exp_rewards, dtype=torch.float32)

        states = State.from_list([states, exp_states])
        actions = Action.from_list([actions, exp_actions])
        rewards = torch.cat([rewards, exp_rewards], axis=0)
        next_states = State.from_list([next_states, exp_next_states])
        weights = torch.cat([weights, exp_weights], axis=0)

        # shuffle tensors
        index = torch.randperm(len(rewards))

        return (states[index],
                actions[index],
                rewards[index],
                next_states[index],
                weights[index])
