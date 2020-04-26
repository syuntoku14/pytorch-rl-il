import logging
from rlil.environments import State
from rlil.initializer import get_logger, get_writer
from rlil.samplers import AsyncSampler, StartInfo
import numpy as np
import torch
import warnings
import os
import time
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
            max_sample_frames=np.inf,
            max_sample_episodes=np.inf,
            max_train_steps=np.inf,
    ):
        self._agent = agent
        self._sampler = sampler
        self._eval_sampler = eval_sampler
        self._max_sample_frames = max_sample_frames
        self._max_sample_episodes = max_sample_episodes
        self._max_train_steps = max_train_steps
        self._writer = get_writer()
        self._logger = get_logger()
        self._best_returns = -np.inf

    def start_training(self):
        while not self._done():
            # training
            start_time = time.time()
            train_steps = self._writer.train_steps

            # sampling for training
            if self._sampler is not None:
                lazy_agent = self._agent.make_lazy_agent()
                self._sampler.start_sampling(lazy_agent,
                                             start_info=self._get_current_info(),
                                             worker_episodes=1)

                sample_result = self._sampler.store_samples(timeout=0.05)

                for sample_info in sample_result.values():
                    self._writer.sample_frames += sum(sample_info["frames"])
                    self._writer.sample_episodes += len(sample_info["frames"])
                    # trains by number of obtained samples
                    for _ in range(sum(sample_info["frames"])):
                        self._agent.train()

            self._agent.train()
            training_msg = {"training time [sec]": round(time.time() - start_time, 2),
                            "trained steps": self._writer.train_steps - train_steps}
            self._logger.info("\nTraining:\n" +
                              json.dumps(training_msg, indent=2))

            # evaluation
            if self._eval_sampler is not None:
                eval_lazy_agent = self._agent.make_lazy_agent(
                    evaluation=True, store_samples=False)
                self._eval_sampler.start_sampling(
                    eval_lazy_agent,
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
                    "train_steps": start_info.train_steps
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
            'train_step', self._writer.train_steps, step="sample_frame")
        self._writer.add_scalar(
            'sample_frame', self._writer.sample_frames, step="train_step")

    def _add_scalar_all(self, name, value, start_info):
        self._writer.add_scalar(name, value,
                                step="sample_episode",
                                step_value=start_info.sample_episodes)
        self._writer.add_scalar(name, value,
                                step="sample_frame",
                                step_value=start_info.sample_frames)
        self._writer.add_scalar(name, value,
                                step="train_step",
                                step_value=start_info.train_steps)

    def _get_current_info(self):
        return StartInfo(sample_frames=self._writer.sample_frames,
                         sample_episodes=self._writer.sample_episodes,
                         train_steps=self._writer.train_steps)

    def _done(self):
        return (
            self._writer.sample_frames > self._max_sample_frames or
            self._writer.sample_episodes > self._max_sample_episodes or
            self._writer.train_steps > self._max_train_steps
        )
