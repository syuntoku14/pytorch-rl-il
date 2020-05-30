import torch
from .abstract import TargetNetwork


class TrivialTarget(TargetNetwork):
    def __init__(self):
        self._target = None

    def __call__(self, *inputs):
        with torch.no_grad():
            return self._target(*inputs)

    def init(self, model):
        self._target = model

    def update(self):
        pass
