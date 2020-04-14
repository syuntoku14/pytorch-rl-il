import pytest
import gym
import torch
import numpy as np
from rlil.environments import Action, State
import rlil.initializer as init


def test_create_state_debug(benchmark):
    init.enable_debug_mode()
    assert init.is_debug_mode()

    raw = torch.randn(3, 4)
    benchmark.pedantic(State,
                       kwargs={'raw': raw},
                       rounds=100,
                       iterations=5)


def test_create_state(benchmark):
    init.disable_debug_mode()
    assert not init.is_debug_mode()

    raw = torch.randn(3, 4)
    benchmark.pedantic(State,
                       kwargs={'raw': raw},
                       rounds=100,
                       iterations=5)
