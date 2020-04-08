import pytest
import numpy as np
from rlil.utils.optim import Schedulable, LinearScheduler


class Obj(Schedulable):
    def __init__(self):
        self.attr = 0


def test_linear_scheduler():
    obj = Obj()
    obj.attr = LinearScheduler(10, 0, 3, 13)
    expected = [10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0]
    actual = [obj.attr for _ in expected]
    np.testing.assert_allclose(actual, expected)
