from abc import ABC, abstractmethod
import numpy as np
import torch
from cpprb import ReplayBuffer, create_env_dict, create_before_add_func
from rlil.environments import State, Action
from rlil.utils.optim import Schedulable
from rlil.initializer import get_device, is_debug_mode


def check_inputs_shapes(store):
    def retfunc(self, states, actions, rewards, next_states):
        if states is None:
            return None
        # device check
        assert states.device == torch.device("cpu"), \
            "Input states.device must be cpu"
        assert actions.device == torch.device("cpu"), \
            "Input actions.device must be cpu"
        assert rewards.device == torch.device("cpu"), \
            "Input rewards.device must be cpu"
        assert next_states.device == torch.device("cpu"), \
            "Input next_states.device must be cpu"

        # type check
        assert isinstance(
            states, State), "Input invalid states type {}. states must be all.environments.State".format(type(states))
        assert isinstance(
            actions, Action), "Input invalid states type {}. actions must be all.environments.Action".format(type(actions))
        assert isinstance(next_states, State), "Input invalid next_states type {}. next_states must be all.environments.State".format(
            type(next_states))
        assert isinstance(
            rewards, torch.FloatTensor), "Input invalid rewards type {}. rewards must be torch.FloatTensor".format(type(rewards))

        # shape check
        assert len(rewards.shape) == 1, "rewards.shape {} must be 'shape == (batch_size)'".format(
            rewards.shape)

        return store(self, states, actions, rewards, next_states)
    return retfunc


class BaseReplayBuffer(ABC):
    @abstractmethod
    def store(self, states, actions, rewards, next_states):
        """Store the transition in the buffer
        Args:
            states (rlil.environment.State): batch_size x shape
            actions (rlil.environment.Action): batch_size x shape
            rewards (torch.Tensor): batch_size
            next_states (rlil.environment.State): batch_size x shape
        """

    @abstractmethod
    def sample(self, batch_size):
        '''Sample from the stored transitions'''

    @abstractmethod
    def update_priorities(self, indexes, td_errors):
        '''Update priorities based on the TD error'''


# TODO: support tuple observation
class ExperienceReplayBuffer(BaseReplayBuffer):
    def __init__(self, size, env):
        self.buffer = ReplayBuffer(size, create_env_dict(env))
        self.before_add = create_before_add_func(env)
        self.device = get_device()

    @check_inputs_shapes
    def store(self, states, actions, rewards, next_states):
        """Store the transition in the buffer
        Args:
            states (rlil.environment.State): batch_size x shape
            actions (rlil.environment.Action): batch_size x shape
            rewards (torch.Tensor): batch_size
            next_states (rlil.environment.State): batch_size x shape
        """

        np_states, np_dones = states.raw_numpy()
        np_actions = actions.raw_numpy()
        np_rewards = rewards.detach().cpu().numpy()
        np_next_states, np_next_dones = next_states.raw_numpy()

        for state, action, reward, next_state, done, next_done in \
                zip(np_states, np_actions, np_rewards,
                    np_next_states, np_dones, np_next_dones):
            if done:
                continue
            self.buffer.add(**self.before_add(obs=state,
                                              act=action,
                                              rew=reward,
                                              done=next_done,
                                              next_obs=next_state))

    def sample(self, batch_size):
        '''Sample from the stored transitions'''
        npsamples = self.buffer.sample(batch_size)
        samples = self.samples_from_np(npsamples)
        weights = torch.ones(batch_size).to(self.device)

        return (*samples, weights)
    
    def update_priorities(self, indexes, td_errors):
        '''Update priorities based on the TD error'''
        pass

    def get_all_transitions(self):
        npsamples = self.buffer.get_all_transitions()
        return self.samples_from_np(npsamples)

    def samples_from_np(self, npsamples, device=None):
        device = self.device if device is None else device

        states = State.from_numpy(npsamples["obs"]).to(device)
        actions = Action.from_numpy(npsamples["act"]).to(device)
        rewards = torch.FloatTensor(npsamples["rew"]).squeeze().to(device)
        next_states = State.from_numpy(
            npsamples["next_obs"], npsamples["done"]).to(device)
        return states, actions, rewards, next_states

    def __len__(self):
        return self.buffer.get_stored_size()
