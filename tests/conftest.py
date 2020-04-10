import pytest
import numpy as np
import torch
from rlil.initializer import set_seed


@pytest.fixture(scope="session", autouse=True)
def seed():
    """set random seed for testing"""
    set_seed(0)
