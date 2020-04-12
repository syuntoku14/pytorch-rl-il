import csv
import os
import subprocess
import torch
import numpy as np
from abc import ABC, abstractmethod
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict


class Writer(ABC):
    log_dir = "runs"

    @abstractmethod
    def add_scalar(self, name, value, step="sample_frame",
                   step_value=None, save_csv=False):
        pass

    @abstractmethod
    def add_text(self, name, text, step="sample_frame"):
        pass

    def _get_step(self, _type):
        if type(_type) is not str:
            raise ValueError("step must be str")
        if _type == "sample_frame":
            return self.sample_frames
        if _type == "sample_episode":
            return self.sample_episodes
        if _type == "train_frame":
            return self.train_frames
        return _type


class DummyWriter(Writer):
    def __init__(self):
        self.sample_frames = 0
        self.sample_episodes = 0
        self.train_frames = 0

    def add_scalar(self, name, value, step="sample_frame",
                   step_value=None, save_csv=False):
        pass

    def add_text(self, name, text, step="sample_frame"):
        pass


class ExperimentWriter(SummaryWriter, Writer):
    def __init__(self, agent_name, env_name,
                 sample_frame_interval=1e4,
                 sample_episode_interval=1e3,
                 train_frame_interval=1e4,
                 exp_info="default_experiments"):

        self.env_name = env_name
        self._add_scalar_interval = \
            {"sample_frame": sample_frame_interval,
             "sample_episode": sample_episode_interval,
             "train_frame": train_frame_interval}

        # make experiment directory
        current_time = str(datetime.now())
        self.log_dir = os.path.join(
            "runs", exp_info, env_name,
            ("%s %s %s" % (agent_name, COMMIT_HASH, current_time))
        )
        self.log_dir = self.log_dir.replace(" ", "_")
        os.makedirs(self.log_dir)

        self.sample_frames = 0
        self.train_frames = 0
        self.sample_episodes = 0
        self._name_frame_history = defaultdict(lambda: 0)
        super().__init__(log_dir=self.log_dir)

    def add_scalar(self, name, value, step="sample_frame",
                   step_value=None, save_csv=False):
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().item()
        if isinstance(value, np.ndarray):
            value = value.item()

        value_name = self.env_name + "/" + name + "/" + step

        # add data every self._add_scalar_interval
        if self._get_step(step) - self._name_frame_history[value_name] >= self._add_scalar_interval[step]:
            step_value = self._get_step(step) if step_value is None else step_value
            super().add_scalar(value_name, value, step_value)
            self._name_frame_history[value_name] = step_value

            if save_csv:
                with open(os.path.join(self.log_dir, name + ".csv"), "a") as csvfile:
                    csv.writer(csvfile).writerow(
                        [self._get_step(step), value])

    def add_text(self, name, text, step="sample_frame"):
        name = self.env_name + "/" + name
        super().add_text(name, text, self._get_step(step))


def get_commit_hash():
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False
    )
    return result.stdout.decode("utf-8").rstrip()


COMMIT_HASH = get_commit_hash()

try:
    os.mkdir("runs")
except FileExistsError:
    pass
