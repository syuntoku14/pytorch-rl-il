import pytest
import numpy as np
import torch
from rlil.initializer import set_seed, enable_debug_mode, set_device


@pytest.fixture(scope="session", autouse=True)
def seed():
    """set random seed for testing"""
    set_seed(0)


@pytest.fixture(scope="session", autouse=True)
def debug():
    enable_debug_mode()


@pytest.fixture
def use_cpu():
    set_device("cpu")


@pytest.fixture
def use_gpu():
    set_device("cuda")
