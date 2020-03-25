import torch
import logging
from .writer import DummyWriter


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
