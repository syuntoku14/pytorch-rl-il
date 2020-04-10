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
    def add_loss(self, name, value, step="sample_frame"):
        pass

    @abstractmethod
    def add_evaluation(self, name, value, step="sample_frame"):
        pass

    @abstractmethod
    def add_scalar(self, name, value, step="sample_frame"):
        pass

    @abstractmethod
    def add_schedule(self, name, value, step="sample_frame"):
        pass

    @abstractmethod
    def add_summary(self, name, mean, std, step="sample_frame"):
        pass

    @abstractmethod
    def add_text(self, name, text, step="sample_frame"):
        pass

    def _get_step(self, _type):
        if _type == "sample_frame":
            return self.sample_frames
        if _type == "sample_episode":
            return self.sample_episodes
        if _type == "train_frame":
            return self.train_frames
        return _type

    @property
    def sample_frames(self):
        return self._sample_frames

    @sample_frames.setter
    def sample_frames(self, frames):
        self._sample_frames = frames

    @property
    def sample_episodes(self):
        return self._sample_episodes

    @sample_episodes.setter
    def sample_episodes(self, episodes):
        self._sample_episodes = episodes

    @property
    def train_frames(self):
        return self._train_frames

    @train_frames.setter
    def train_frames(self, frames):
        self._train_frames = frames


class DummyWriter(Writer):
    def __init__(self):
        self._sample_frames = 0
        self._sample_episodes = 1
        self._train_frames = 0

    def add_scalar(self, key, value, step="sample_frame"):
        pass

    def add_loss(self, name, value, step="sample_frame"):
        pass

    def add_schedule(self, name, value, step="sample_frame"):
        pass

    def add_evaluation(self, name, value, step="sample_frame"):
        pass

    def add_summary(self, name, mean, std, step="sample_frame"):
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

        self._sample_frames = 0
        self._train_frames = 0
        self._sample_episodes = 1
        self._name_frame_history = defaultdict(lambda: 0)
        super().__init__(log_dir=self.log_dir)

    def add_loss(self, name, value, step="sample_frame"):
        self.add_scalar("loss/" + name, value, step)

    def add_evaluation(self, name, value, step="sample_frame"):
        self.add_scalar("evaluation/" + name, value, step)

    def add_schedule(self, name, value, step="sample_frame"):
        self.add_scalar("schedule" + "/" + name, value, step)

    def add_scalar(self, name, value, step="sample_frame"):
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().item()
        if isinstance(value, np.ndarray):
            value = value.item()

        if type(step) is str:
            name = self.env_name + "/" + name + "/" + step
        else:
            name = self.env_name + "/" + name + "/unknown"

        # add data every self._add_scalar_interval
        if self._get_step(step) - self._name_frame_history[name] > self._add_scalar_interval[step]:
            super().add_scalar(name, value, self._get_step(step))
            self._name_frame_history[name] = self._get_step(step)

    def add_summary(self, name, mean, std, step="sample_frame"):
        self.add_evaluation(name + "/mean", mean, step)
        self.add_evaluation(name + "/std", std, step)

        with open(os.path.join(self.log_dir, name + ".csv"), "a") as csvfile:
            csv.writer(csvfile).writerow([self._get_step(step), mean, std])

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
