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
    def add_loss(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_evaluation(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_scalar(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_schedule(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_summary(self, name, mean, std, step="frame"):
        pass

    @abstractmethod
    def add_text(self, name, text, step="frame"):
        pass


class DummyWriter(Writer):
    def __init__(self):
        self._frames = 0
        self._episodes = 1

    def add_scalar(self, key, value, step="frame"):
        pass

    def add_loss(self, name, value, step="frame"):
        pass

    def add_schedule(self, name, value, step="frame"):
        pass

    def add_evaluation(self, name, value, step="frame"):
        pass

    def add_summary(self, name, mean, std, step="frame"):
        pass

    def add_text(self, name, text, step="frame"):
        pass

    def _get_step(self, _type):
        if _type == "frame":
            return self.frames
        if _type == "episode":
            return self.episodes
        return _type

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, frames):
        self._frames = frames

    @property
    def episodes(self):
        return self._episodes

    @episodes.setter
    def episodes(self, episodes):
        self._episodes = episodes


class ExperimentWriter(SummaryWriter, Writer):
    def __init__(self, agent_name, env_name,
                 interval=1e4, exp_info="default_experiments"):
        self.env_name = env_name
        current_time = str(datetime.now())
        self.log_dir = os.path.join(
            "runs", exp_info, env_name,
            ("%s %s %s" % (agent_name, COMMIT_HASH, current_time))
        )
        self.log_dir = self.log_dir.replace(" ", "_")
        os.makedirs(self.log_dir)
        self._frames = 0
        self._train_iters = 0
        self._episodes = 1
        self._name_frame_history = defaultdict(lambda: 0)
        self._add_scalar_interval = interval
        super().__init__(log_dir=self.log_dir)

    def add_loss(self, name, value, step="frame"):
        self.add_scalar("loss/" + name, value, step)

    def add_evaluation(self, name, value, step="frame"):
        self.add_scalar("evaluation/" + name, value, self._get_step(step))

    def add_schedule(self, name, value, step="frame"):
        self.add_scalar("schedule" + "/" + name,
                        value, self._get_step(step))

    def add_scalar(self, name, value, step="frame"):
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().item()
        if isinstance(value, np.ndarray):
            value = value.item()

        name = self.env_name + "/" + name
        # add data every self._add_scalar_interval
        if self._get_step("frame") - self._name_frame_history[name] > self._add_scalar_interval:
            super().add_scalar(name, value, self._get_step(step))
            self._name_frame_history[name] = self._get_step("frame")

    def add_summary(self, name, mean, std, step="frame"):
        self.add_evaluation(name + "/mean", mean, step)
        self.add_evaluation(name + "/std", std, step)

        with open(os.path.join(self.log_dir, name + ".csv"), "a") as csvfile:
            csv.writer(csvfile).writerow([self._get_step(step), mean, std])

    def add_text(self, name, text, step="frame"):
        name = self.env_name + "/" + name
        super().add_text(name, text, self._get_step(step))

    def _get_step(self, _type):
        if _type == "frame":
            return self.frames
        if _type == "episode":
            return self.episodes
        if _type == "train_iters":
            return self.train_iters
        return _type

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, frames):
        self._frames = frames

    @property
    def episodes(self):
        return self._episodes

    @episodes.setter
    def episodes(self, episodes):
        self._episodes = episodes

    @property
    def train_iters(self):
        return self._train_iters

    @train_iters.setter
    def train_iters(self, train_iters):
        self._train_iters = train_iters


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
