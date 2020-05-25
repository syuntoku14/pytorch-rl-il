import warnings
from abc import abstractmethod, ABC
import torch
import os
from rlil.initializer import get_writer


class Checkpointer(ABC):
    @abstractmethod
    def init(self, model, filename):
        pass

    @abstractmethod
    def __call__(self):
        pass


class DummyCheckpointer(Checkpointer):
    def init(self, *inputs):
        pass

    def __call__(self):
        pass


class PeriodicCheckpointer(Checkpointer):
    def __init__(self, frequency):
        self.frequency = frequency
        self._writer = get_writer()
        self._log_dir = None
        self._filename = None
        self._model = None

    def init(self, model, log_dir, filename):
        self._model = model
        self._log_dir = log_dir
        self._filename = filename
        # Some builds of pytorch throw this unhelpful warning.
        # We can safely disable it.
        # https://discuss.pytorch.org/t/got-warning-couldnt-retrieve-source-code-for-container/7689/7
        warnings.filterwarnings(
            "ignore", message="Couldn't retrieve source code")

    def __call__(self):
        # save pereodically
        # if self._writer.train_steps % self.frequency == 0:
        #     save_dir = os.path.join(self._log_dir, str(self._writer.train_steps))
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
        #     torch.save(self._model, os.path.join(
        #         save_dir, self._filename) + ".pt")

        if self._writer.train_steps % self.frequency == 0:
            torch.save(self._model,
                       os.path.join(self._log_dir, self._filename + ".pt"))
