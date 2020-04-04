import logging
from rlil.environments import State
from rlil.initializer import get_logger, get_writer
import numpy as np
import torch
import warnings
import os
from abc import ABC, abstractmethod
from timeit import default_timer as timer


class OfflineTrainer(ABC):
    def __init__(
            self,
            agent_fn,
            env,
            writer,
            seed=0,
            iters=np.inf,
            eval_intervals=1e3,
            render=False,
    ):
        self._agent = agent_fn(env)
        self._env = env
        self._writer = get_writer()
        self._max_iters = iters
        self._render = render
        self._best_returns = -np.inf
        self._returns100 = []
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._env.seed(seed)
        self._logger = get_logger()

        self.run()

    @abstractmethod
    def train(self):
        pass

    def _done(self):
        return self._writer.train_iters > self._max_iters

    def _log(self, returns):
        self._logger.info("train_iters: %d, returns: %d" %
                            (self._writer.train_iters, returns))
        if returns > self._best_returns:
            self._best_returns = returns
        self._returns100.append(returns)
        if len(self._returns100) == 100:
            mean = np.mean(self._returns100)
            std = np.std(self._returns100)
            self._writer.add_summary('returns100', mean, std, step="train_iters")
            self._returns100 = []
        self._writer.add_evaluation('returns/train_iters', returns, step="train_iters")
        self._writer.add_evaluation(
            "returns/max", self._best_returns, step="train_iters")


class SingleEnvTrainer(BatchTrainer):
    def run(self):
        while not self._done():
            self._run_episode()

    def _run_episode(self):
        start_time = timer()
        start_frames = self._writer.frames
        returns = self._run_until_terminal_state()
        end_time = timer()
        fps = (self._writer.frames - start_frames) / (end_time - start_time)
        self._log(returns, fps)
        self._writer.episodes += 1

    def _run_until_terminal_state(self):
        agent = self._agent
        env = self._env

        env.reset()
        returns = 0
        action = agent.act(env.state, env.reward)

        while not env.done:
            self._writer.frames += 1
            if self._render:
                env.render()
            env.step(action)
            returns += env.reward
            action = agent.act(env.state, env.reward)

        return returns

