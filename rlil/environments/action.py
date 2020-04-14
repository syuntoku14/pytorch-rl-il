import numpy as np
import torch
import warnings
import gym
from rlil.initializer import is_debug_mode, get_device


def action_decorator(func):
    def retfunc(*args, **kwargs):
        action = func(*args, **kwargs)
        action = Action(action)
        return action
    return retfunc


def clip_action(raw, low, high):
    actions = torch.min(raw, high)
    actions = torch.max(actions, low)
    return actions


def squash_action(raw, tanh_scale=None, tanh_mean=None, action_space=None):
    """squash_action converts the raw value into valid actions using tanh
    This function requires a pair of (tanh_scale, tanh_mean) or an action_space

    Parameters
    ----------
    raw : torch.FloatTensor
    tanh_scale : torch.FloatTensor
    tanh_mean : torch.FloatTensor
    action_space : gym.spaces.Box

    Returns
    -------
    actions : torch.FloatTensor
    """
    if tanh_scale is None or tanh_mean is None:
        tanh_scale = torch.tensor(
            (action_space.high - action_space.low) / 2).to(raw.device)
        tanh_mean = torch.tensor(
            (action_space.high + action_space.low) / 2).to(raw.device)

    actions = torch.tanh(raw) * tanh_scale + tanh_mean
    return actions


class Action:
    _action_space = None

    def __init__(self, raw):
        """
        If is_debug_mode()==True,
            1. Detecting invalid tensor shape. The raw (torch.Tensor) must be batch_size x shape
            2. Detecting invalid discrete action value
            3. Cliping continuous action value
        """

        if is_debug_mode():
            assert self._action_space is not None, \
                "action_space is not set. Use Action.set_action_space function."
            assert isinstance(raw, torch.Tensor), \
                "Input invalid raw type {}. raw must be torch.Tensor".format(
                    type(raw))
            assert len(raw.shape) > 1, \
                "Action.raw.shape {} is invalid. Batch_size must be specified".format(
                    raw.shape)

            if isinstance(self._action_space, gym.spaces.Discrete):
                assert raw.shape[1] == 1, \
                    "Action.raw.shape {} is invalid. Discrete action's shape must be batch_size x 1".format(
                        raw.shape)
                assert (0 <= raw).all() and (
                    raw < self._action_space.n).all(), "Invalid action value"
            elif isinstance(self._action_space, gym.spaces.Box):
                assert raw.shape[1:] == self._action_space.shape, \
                    "Action.raw.shape {} is invalid. It doesn't match the action_space.".format(
                        raw.shape)
            else:
                raise TypeError("Unknown action space type")

        self._raw = raw

    @classmethod
    def set_action_space(cls, action_space):
        assert isinstance(action_space, gym.spaces.Discrete) or isinstance(
            action_space, gym.spaces.Box), "Invalid action space"
        cls._action_space = action_space
        device = get_device()
        if isinstance(action_space, gym.spaces.Box):
            cls._low = torch.tensor(action_space.low, device=device)
            cls._high = torch.tensor(action_space.high, device=device)

    @classmethod
    def from_numpy(cls, actions):
        raw = torch.from_numpy(actions)
        return cls(raw)

    @classmethod
    def from_list(cls, actions):
        raw = torch.cat([action.raw for action in actions])
        return cls(raw)

    @classmethod
    def action_space(cls):
        assert cls._action_space is not None, "action_space is not set. Use Action.set_action_space function."
        return cls._action_space

    @property
    def features(self):
        '''
        Convert self._raw into valid value by action_space
        '''
        if isinstance(self._action_space, gym.spaces.Discrete):
            return self._raw
        if isinstance(self._action_space, gym.spaces.Box):
            # clip the action into the valid range
            return clip_action(self._raw, self._low.to(self.device), self._high.to(self.device))

    @property
    def raw(self):
        return self._raw

    @property
    def device(self):
        return self._raw.device

    def to(self, device):
        return Action(
            self._raw.to(device),
        )

    def detach(self):
        self._raw.detach()

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return Action(
                self._raw[idx]
            )
        if isinstance(idx, torch.Tensor):
            return Action(
                self._raw[idx]
            )
        return Action(
            self._raw[idx].unsqueeze(0)
        )

    def __len__(self):
        return len(self._raw)
