import os
import torch.multiprocessing as mp
import torch
import logging
from rlil.utils.writer import DummyWriter

os.environ["PYTHONWARNINGS"] = 'ignore:semaphore_tracker:UserWarning'

_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_device(device):
    global _DEVICE
    _DEVICE = device


def get_device():
    return _DEVICE


_WRITER = DummyWriter()


def set_writer(writer):
    global _WRITER
    _WRITER = writer


def get_writer():
    return _WRITER


_LOGGER = logging.getLogger(__name__)


def set_logger(logger):
    global _LOGGER
    _LOGGER = logger


def get_logger():
    return _LOGGER


_REPLAY_BUFFER = None


def set_replay_buffer(replay_buffer):
    global _REPLAY_BUFFER
    _REPLAY_BUFFER = replay_buffer


def get_replay_buffer():
    global _REPLAY_BUFFER
    if _REPLAY_BUFFER is None:
        raise ValueError("replay_buffer is not set")
    return _REPLAY_BUFFER
