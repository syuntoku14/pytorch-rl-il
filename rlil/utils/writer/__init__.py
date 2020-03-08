import csv
import os
import subprocess
import torch
from abc import ABC, abstractmethod
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


def value_decorator(func):
    def retfunc(self, name, value, step="frame"):
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().item()
        func(self, name, value, step)
    return retfunc


def summary_decorator(func):
    def retfunc(self, name, mean, std, step="frame"):
        if isinstance(mean, torch.Tensor):
            mean = mean.cpu().detach().item()
        if isinstance(std, torch.Tensor):
            std = std.cpu().detach().item()
        func(self, name, mean, std, step)
    return retfunc


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


class DummyWriter(Writer):
    def __init__(self):
        self.frames = 0
        self.episodes = 1

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

    def _get_step(self, _type):
        if _type == "frame":
            return self.frames
        if _type == "episode":
            return self.episodes
        return _type


class ExperimentWriter(SummaryWriter, Writer):
    def __init__(self, agent_name, env_name, loss=True):
        self.env_name = env_name
        current_time = str(datetime.now())
        self.log_dir = os.path.join(
            "runs", ("%s_%s_%s" % (agent_name, COMMIT_HASH, current_time))
        )
        self.log_dir = self.log_dir.replace(" ", "_")
        os.makedirs(os.path.join(self.log_dir, env_name))
        self._frames = 0
        self._episodes = 1
        self._loss = loss
        super().__init__(log_dir=self.log_dir)

    @value_decorator
    def add_loss(self, name, value, step="frame"):
        if self._loss:
            self.add_scalar("loss/" + name, value, step)

    @value_decorator
    def add_evaluation(self, name, value, step="frame"):
        self.add_scalar("evaluation/" + name, value, self._get_step(step))

    @value_decorator
    def add_schedule(self, name, value, step="frame"):
        if self._loss:
            self.add_scalar("schedule" + "/" + name, value, self._get_step(step))

    @value_decorator
    def add_scalar(self, name, value, step="frame"):
        super().add_scalar(self.env_name + "/" + name, value, self._get_step(step))

    @summary_decorator
    def add_summary(self, name, mean, std, step="frame"):
        self.add_evaluation(name + "/mean", mean, step)
        self.add_evaluation(name + "/std", std, step)

        with open(os.path.join(self.log_dir, self.env_name, name + ".csv"), "a") as csvfile:
            csv.writer(csvfile).writerow([self._get_step(step), mean, std])

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
