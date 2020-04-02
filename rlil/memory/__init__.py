from .replay_buffer import (
    ReplayBuffer,
    ExperienceReplayBuffer,
    PrioritizedReplayBuffer,
    NStepReplayBuffer,
)

__all__ = [
    "ReplayBuffer",
    "ExperienceReplayBuffer",
    "PrioritizedReplayBuffer",
    "NStepReplayBuffer",
]

_REPLAY_BUFFER = None


def set_replay_buffer(replay_buffer):
    global _REPLAY_BUFFER
    _REPLAY_BUFFER = replay_buffer


def get_replay_buffer():
    global _REPLAY_BUFFER
    if _REPLAY_BUFFER is None:
        raise ValueError("replay_buffer is not set")
    return _REPLAY_BUFFER
