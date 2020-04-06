import logging
from rlil.environments import State
from rlil.initializer import get_logger, get_writer
import numpy as np
import torch
import warnings
import os
from timeit import default_timer as timer


class Trainer:
    """ 
    Trainer trains the agent with given an env and a sampler.
    """
    def __init__(
            self,
            agent_fn,
            env,
            sampler_class,
            seed=0,
            max_frames=np.inf,
            max_episodes=np.inf,
    ):
        self._agent = agent_fn(env)
        self._env = env
        self._writer = get_writer()
        self._max_frames = max_frames
        self._max_episodes = max_episodes
        self._logger = get_logger()

        self._best_returns = -np.inf
        self._returns100 = []

    def start_training(self):
        while not self._done():
            self._run_episode()

    def _done(self):
        return (
            self._writer.frames > self._max_frames or
            self._writer.episodes > self._max_episodes
        )

    def _log(self, returns, fps):
        self._logger.info("episode: %i, frames: %i, fps: %d, returns: %d" %
                          (self._writer.episodes, self._writer.frames, fps, returns))
        if returns > self._best_returns:
            self._best_returns = returns
        self._returns100.append(returns)
        if len(self._returns100) == 100:
            mean = np.mean(self._returns100)
            std = np.std(self._returns100)
            self._writer.add_summary('returns100', mean, std, step="frame")
            self._returns100 = []
        self._writer.add_evaluation('returns/episode', returns, step="episode")
        self._writer.add_evaluation('returns/frame', returns, step="frame")
        self._writer.add_evaluation(
            "returns/max", self._best_returns, step="frame")
        self._writer.add_scalar('fps', fps, step="frame")

