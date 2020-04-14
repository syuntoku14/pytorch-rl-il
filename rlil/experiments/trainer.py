import logging
from rlil.environments import State
from rlil.initializer import get_logger, get_writer
from rlil.samplers import AsyncSampler, StartInfo
import numpy as np
import torch
import warnings
import os
from timeit import default_timer as timer
import json


class Trainer:
    """
    Trainer trains the agent with an env and a sampler.
    """

    def __init__(
            self,
            agent,
            sampler=None,
            eval_sampler=None,
            max_frames=np.inf,
            max_episodes=np.inf,
            num_trains_per_iter=100,
    ):
        self._agent = agent
        self._sampler = sampler
        self._eval_sampler = eval_sampler
        self._max_frames = max_frames
        self._max_episodes = max_episodes
        self._writer = get_writer()
        self._logger = get_logger()
        self._best_returns = -np.inf
        self._num_trains = num_trains_per_iter

    def start_training(self):
        while not self._done():
            # training
            if self._sampler is not None:
                lazy_agent = self._agent.make_lazy_agent()
                self._sampler.start_sampling(lazy_agent,
                                             start_info=self._get_current_info(),
                                             worker_episodes=1)

                sample_result = self._sampler.store_samples(timeout=0.05)

                for sample_info in sample_result.values():
                    self._writer.sample_frames += sum(sample_info["frames"])
                    self._writer.sample_episodes += len(sample_info["frames"])

            for _ in range(self._num_trains):
                self._agent.train()

            # evaluation
            if self._eval_sampler is not None:
                eval_lazy_agent = self._agent.make_lazy_agent(evaluation=True)
                self._eval_sampler.start_sampling(
                    lazy_agent,
                    start_info=self._get_current_info(),
                    worker_episodes=10)
                eval_sample_result = self._eval_sampler.store_samples(
                    timeout=0.05, evaluation=True)

                for start_info, sample_info in eval_sample_result.items():
                    self._log(start_info, sample_info)

    def _log(self, start_info, sample_info):
        mean_returns = np.mean(sample_info["returns"])
        evaluation_msg = \
            {
                "Conditions":
                {
                    "sample_frames": start_info.sample_frames,
                    "sample_episodes": start_info.sample_episodes,
                    "train_frames": start_info.train_frames
                },
                "Result":
                {
                    "collected_frames": sum(sample_info["frames"]),
                    "collected_episodes": len(sample_info["frames"]),
                    "mean returns": round(mean_returns, 2)
                }
            }
        self._logger.info("\nEvaluation:\n" +
                          json.dumps(evaluation_msg, indent=2))

        # update best_returns
        self._best_returns = max(max(sample_info["returns"]),
                                 self._best_returns)

        # log raw returns
        self._add_scalar_all("evaluation/returns", mean_returns, start_info)
        self._add_scalar_all("evaluation/returns/max",
                             self._best_returns, start_info)

        # log sample and train ratio
        self._writer.add_scalar(
            'train_frame', self._writer.train_frames, step="sample_frame")
        self._writer.add_scalar(
            'sample_frame', self._writer.sample_frames, step="train_frame")

    def _add_scalar_all(self, name, value, start_info):
        self._writer.add_scalar(name, value,
                                step="sample_episode",
                                step_value=start_info.sample_episodes)
        self._writer.add_scalar(name, value,
                                step="sample_frame",
                                step_value=start_info.sample_frames)
        self._writer.add_scalar(name, value,
                                step="train_frame",
                                step_value=start_info.train_frames)

    def _get_current_info(self):
        return StartInfo(sample_frames=self._writer.sample_frames,
                         sample_episodes=self._writer.sample_episodes,
                         train_frames=self._writer.train_frames)

    def _done(self):
        return (
            self._writer.sample_frames > self._max_frames or
            self._writer.sample_episodes > self._max_episodes
        )
