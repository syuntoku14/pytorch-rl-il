import pytest
import numpy as np
import torch


@pytest.fixture(scope="session", autouse=True)
def seed():
    """set random seed for testing"""
    np.random.seed(0)
    torch.manual_seed(0)
