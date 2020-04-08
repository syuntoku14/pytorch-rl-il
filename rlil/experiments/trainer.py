import logging
from rlil.environments import State
from rlil.initializer import get_logger, get_writer
import numpy as np
import torch
import warnings
import os
from timeit import default_timer as timer
from rlil.initializer import get_writer, get_logger


class Trainer:
    """ 
    Trainer trains the agent with given an env and a sampler.
    """

    def __init__(
            self,
            agent,
            sampler,
            max_frames=np.inf,
            max_episodes=np.inf,
    ):
        self._agent = agent
        self._sampler = sampler
        self._max_frames = max_frames
        self._max_episodes = max_episodes
        self._writer = get_writer()
        self._logger = get_logger()
        self._best_returns = -np.inf
        self._returns100 = []

    def start_training(self):
        while not self._done():
            lazy_agent = self._agent.make_lazy_agent()
            self._sampler.start_sampling(
                lazy_agent, worker_episodes=1)
            sample_info = self._sampler.store_samples(timeout=0.05)
            self._writer.sample_frames += sample_info["frames"]
            self._writer.sample_episodes += sample_info["episodes"]
            for _ in range(int(sample_info["frames"] / len(self._sampler._workers))):
                self._agent.train()
            for returns in sample_info["returns"]:
                self._log(returns.item())

    def _done(self):
        return (
            self._writer.sample_frames > self._max_frames or
            self._writer.sample_episodes > self._max_episodes
        )

    def _log(self, returns):
        self._logger.info("episode: %i, frames: %i, returns: %d" %
                          (self._writer.sample_episodes,
                           self._writer.sample_frames, returns))
        if returns > self._best_returns:
            self._best_returns = returns
        self._returns100.append(returns)
        if len(self._returns100) == 100:
            mean = np.mean(self._returns100)
            std = np.std(self._returns100)
            self._writer.add_summary('returns100', mean, std, step="sample_frame")
            self._returns100 = []
        self._writer.add_evaluation('returns/episode', returns, step="sample_episode")
        self._writer.add_evaluation('returns/frame', returns, step="sample_frame")
        self._writer.add_evaluation(
            "returns/max", self._best_returns, step="sample_frame")
