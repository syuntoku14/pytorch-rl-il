import pytest
import gym
import torch
import numpy as np
from rlil.environments import Action, State
import rlil.initializer as init


@pytest.fixture()
def set_action_space():
    action_space = gym.spaces.Box(
        low=np.array([-1, -10]), high=np.array([1, 10]))
    Action.set_action_space(action_space)

    raw = torch.FloatTensor([[0, 0], [2, 2], [-20, -20]])
    yield raw


def test_create_action_debug(set_action_space,
                             benchmark):
    init.enable_debug_mode()
    assert init.is_debug_mode()

    raw = set_action_space
    action = benchmark.pedantic(Action,
                                kwargs={'raw': raw},
                                rounds=100,
                                iterations=5)


def test_create_action(set_action_space,
                       benchmark):
    init.disable_debug_mode()
    assert not init.is_debug_mode()

    raw = set_action_space
    action = benchmark.pedantic(Action,
                                kwargs={'raw': raw},
                                rounds=100,
                                iterations=5)


def get_features(action):
    return action.features


def test_features_action_cpu(set_action_space,
                             benchmark):
    raw = set_action_space
    action = Action(raw)

    benchmark.pedantic(get_features,
                       rounds=100,
                       kwargs={"action": action},
                       iterations=5)


def test_features_action_cuda(set_action_space,
                              benchmark):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    raw = set_action_space
    action = Action(raw.to("cuda"))

    benchmark.pedantic(get_features,
                       rounds=100,
                       kwargs={"action": action},
                       iterations=5)
