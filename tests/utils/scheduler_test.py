import pytest
import numpy as np
from rlil.utils.scheduler import LinearScheduler


def test_linear_scheduler():
    epsilon = LinearScheduler(10, 0, 10)
    expected = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0]
    actual = []
    for _ in expected:
        actual.append(epsilon.get())
        epsilon.update()
    np.testing.assert_allclose(actual, expected)
