import pytest
import numpy as np
import torch
from rlil.initializer import set_seed, enable_debug_mode, set_device, get_device
from rlil import nn
from rlil.environments import GymEnvironment, Action
from rlil.policies.deterministic import DeterministicPolicyNetwork
from rlil.memory import ExperienceReplayBuffer


@pytest.fixture(scope="function", autouse=True)
def seed():
    """set random seed for testing"""
    set_seed(0)


@pytest.fixture(scope="function", autouse=True)
def debug():
    enable_debug_mode()


@pytest.fixture(scope="function", autouse=True)
def reset_action_space():
    Action._action_space = None


@pytest.fixture
def use_cpu():
    pre_device = get_device()
    set_device("cpu")
    yield
    set_device(pre_device)


@pytest.fixture
def use_gpu():
    pre_device = get_device()
    set_device("cuda")
    yield
    set_device(pre_device)
