import numpy as np
import torch
from cpprb import (ReplayBuffer, PrioritizedReplayBuffer,
                   create_env_dict, create_before_add_func)
from rlil.environments import State, Action
from rlil.initializer import get_device, is_debug_mode
from rlil.utils import Samples, samples_to_np
from .base import BaseReplayBuffer


def check_samples(samples, priorities=None):
    states, actions, rewards, next_states, _, _ = samples

    # type check
    assert isinstance(states, State), "Input invalid states type {}. \
        states must be all.environments.State".format(type(states))
    assert isinstance(actions, Action), "Input invalid states type {}. \
            actions must be all.environments.Action".format(type(actions))
    assert isinstance(next_states, State), \
        "Input invalid next_states type {}. next_states must be all.environments.State".format(
        type(next_states))
    assert isinstance(rewards, torch.Tensor), "Input invalid rewards type {}. \
        rewards must be torch.Tensor".format(type(rewards))
    if priorities is not None:
        assert isinstance(priorities, torch.Tensor), "Input invalid priorities type {}. \
            priorities must be torch.Tensor".format(type(priorities))

    # shape check
    assert len(rewards.shape) == 1, \
        "rewards.shape {} must be 'shape == (batch_size)'".format(
            rewards.shape)
    if priorities is not None:
        assert len(priorities.shape) == 1, \
            "priorities.shape {} must be 'shape == (batch_size)'".format(
                priorities.shape)


def check_inputs_shapes(store):
    def retfunc(self, samples, priorities=None):
        if samples.states is None:
            return None
        if is_debug_mode():
            check_samples(samples, priorities=priorities)
        return store(self, samples, priorities=priorities)
    return retfunc


class ExperienceReplayBuffer(BaseReplayBuffer):
    '''This class utilizes cpprb.ReplayBuffer'''

    def __init__(self,
                 size, env,
                 prioritized=False, alpha=0.6, beta=0.4, eps=1e-4,
                 n_step=1, discount_factor=0.95):
        """
        Args:
            size (int): The capacity of replay buffer.
            env (rlil.environments.GymEnvironment)
            prioritized (bool): Use prioritized replay buffer if True.
            alpha, beta, eps (float): 
                Hyperparameter of PrioritizedReplayBuffer.
                See https://arxiv.org/abs/1511.05952.
            n_step (int, optional):
               Number of steps for Nstep experience replay.
               If n_step > 1, you need to call self.on_episode_end()
               before every self.sample(). The n_step calculation is done
               in LazyAgent objects, not in Agent objects.
            discount_factor (float, optional): 
                Discount factor for Nstep experience replay.
        """

        # common
        self._before_add = create_before_add_func(env)
        self.device = get_device()
        env_dict = create_env_dict(env)

        # Nstep
        Nstep = None
        if n_step > 1:
            Nstep = {"size": n_step, "rew": "rew",
                     "next": "next_obs", "gamma": discount_factor}
        self._n_step = n_step

        # PrioritizedReplayBuffer
        self.prioritized = prioritized
        self._beta = beta
        if prioritized:
            self._buffer = PrioritizedReplayBuffer(size, env_dict,
                                                   alpha=alpha, eps=eps,
                                                   Nstep=Nstep)
        else:
            self._buffer = ReplayBuffer(size, env_dict, Nstep=Nstep)

    @check_inputs_shapes
    def store(self, samples, priorities=None):
        """Store the samples in the buffer
        Args:
            Samples(
                states (rlil.environment.State): batch_size x shape
                actions (rlil.environment.Action): batch_size x shape
                rewards (torch.Tensor): batch_size
                next_states (rlil.environment.State): batch_size x shape
                weights: None
                indexes: None
            )
            priorities (torch.Tensor): batch_size
        """

        np_states, np_rewards, np_actions, np_next_states, \
            np_dones, np_next_dones = samples_to_np(samples)

        assert len(np_states) < self._buffer.get_buffer_size(), \
            "The sample size exceeds the buffer size."

        if self.prioritized and (~np_dones).any():
            np_priorities = None if priorities is None \
                else priorities.detach().cpu().numpy()[~np_dones]
            self._buffer.add(
                **self._before_add(obs=np_states[~np_dones],
                                   act=np_actions[~np_dones],
                                   rew=np_rewards[~np_dones],
                                   done=np_next_dones[~np_dones],
                                   next_obs=np_next_states[~np_dones]),
                priorities=np_priorities)

        # if there is at least one sample to store
        if not self.prioritized and (~np_dones).any():
            # remove done==1 by [~np_dones]
            self._buffer.add(
                **self._before_add(obs=np_states[~np_dones],
                                   act=np_actions[~np_dones],
                                   rew=np_rewards[~np_dones],
                                   done=np_next_dones[~np_dones],
                                   next_obs=np_next_states[~np_dones]))

    def sample(self, batch_size):
        '''Sample from the stored transitions'''
        if self.prioritized:
            npsamples = self._buffer.sample(batch_size, beta=self._beta)
        else:
            npsamples = self._buffer.sample(batch_size)
        samples = self.samples_from_cpprb(npsamples)
        return samples

    def update_priorities(self, indexes, td_errors):
        '''Update priorities based on the TD error'''
        if is_debug_mode():
            # shape check
            assert len(td_errors.shape) == 1, \
                "rewards.shape {} must be 'shape == (batch_size)'".format(
                    rewards.shape)
            assert td_errors.device == torch.device("cpu"), \
                "td_errors must be cpu tensors"

        if self.prioritized:
            self._buffer.update_priorities(indexes, td_errors.detach().numpy())

    def get_all_transitions(self, return_cpprb=False):
        npsamples = self._buffer.get_all_transitions()
        if return_cpprb:
            return npsamples
        return self.samples_from_cpprb(npsamples)

    def samples_from_cpprb(self, npsamples, device=None):
        """
        Convert samples generated by cpprb.ReplayBuffer.sample() into 
        State, Action, rewards, State.
        Return Samples object.

        Args:
            npsamples (dict of nparrays): 
                Samples generated by cpprb.ReplayBuffer.sample()

            device (optional): The device where the outputs are loaded.

        Returns:
            Samples(State, Action, torch.FloatTensor, State)
        """
        device = self.device if device is None else device

        states = State.from_numpy(npsamples["obs"], device=device)
        actions = Action.from_numpy(npsamples["act"], device=device)
        rewards = torch.tensor(npsamples["rew"], dtype=torch.float32,
                               device=device).squeeze()
        next_states = State.from_numpy(
            npsamples["next_obs"], npsamples["done"], device=device)
        if self.prioritized:
            weights = torch.tensor(
                npsamples["weights"], dtype=torch.float32, device=self.device)
            indexes = npsamples["indexes"]
        else:
            weights = torch.ones(states.shape[0], device=self.device)
            indexes = None
        return Samples(states, actions, rewards, next_states, weights, indexes)

    def on_episode_end(self):
        if self._n_step > 1:
            self._buffer.on_episode_end()

    def clear(self):
        self._buffer.clear()

    def __len__(self):
        return self._buffer.get_stored_size()
