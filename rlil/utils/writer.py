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
    def add_scalar(self, name, value, step="sample_frames",
                   step_value=None, save_csv=False):
        pass

    @abstractmethod
    def add_text(self, name, text, step="sample_frames"):
        pass

    def _get_step_value(self, _type):
        if type(_type) is not str:
            raise ValueError("step must be str")
        if _type == "sample_frames":
            return self.sample_frames
        if _type == "sample_episodes":
            return self.sample_episodes
        if _type == "train_steps":
            return self.train_steps
        return _type


class DummyWriter(Writer):
    def __init__(self):
        self.sample_frames = 0
        self.sample_episodes = 0
        self.train_steps = 0

    def add_scalar(self, name, value, step="sample_frames",
                   step_value=None, save_csv=False):
        pass

    def add_text(self, name, text, step="sample_frames"):
        pass


class ExperimentWriter(SummaryWriter, Writer):
    def __init__(self, agent_name, env_name,
                 sample_frame_interval=1e4,
                 sample_episode_interval=1e2,
                 train_step_interval=1e2,
                 exp_info="default_experiments"):
        try:
            os.mkdir("runs")
        except FileExistsError:
            pass
        self.env_name = env_name
        self._add_scalar_interval = \
            {"sample_frames": sample_frame_interval,
             "sample_episodes": sample_episode_interval,
             "train_steps": train_step_interval}

        # make experiment directory
        current_time = str(datetime.now())
        self.log_dir = os.path.join(
            "runs", exp_info, env_name,
            ("%s %s %s" % (agent_name, COMMIT_HASH, current_time))
        )
        self.log_dir = self.log_dir.replace(" ", "_")
        os.makedirs(self.log_dir)

        self.sample_frames = 0
        self.train_steps = 0
        self.sample_episodes = 0
        self._name_frame_history = defaultdict(lambda: 0)
        super().__init__(log_dir=self.log_dir)

    def add_scalar(self, name, value, step="train_steps",
                   step_value=None, save_csv=False):
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().item()
        if isinstance(value, np.ndarray):
            value = value.item()

        step_value = self._get_step_value(
            step) if step_value is None else step_value
        value_name = self.env_name + "/" + name + "/" + step

        # add data every self._add_scalar_interval
        if step_value - self._name_frame_history[value_name] >= self._add_scalar_interval[step]:
            super().add_scalar(value_name, value, step_value)
            self._name_frame_history[value_name] = step_value

            if save_csv:
                with open(os.path.join(self.log_dir, name + ".csv"), "a") as csvfile:
                    csv.writer(csvfile).writerow(
                        [step_value, value])

    def add_text(self, name, text, step="train_steps"):
        name = self.env_name + "/" + name
        super().add_text(name, text, self._get_step_value(step))

    def add_histogram(self, name, values, step="train_steps"):
        # add histogram every self._add_scalar_interval * 100
        step_value = self._get_step_value(step)
        value_name = self.env_name + "/" + name + "/" + step
        if step_value - self._name_frame_history[value_name] \
                >= self._add_scalar_interval[step] * 100:
            super().add_histogram(value_name, values, self._get_step_value(step))
            self._name_frame_history[value_name] = step_value


def get_commit_hash():
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False
    )
    return result.stdout.decode("utf-8").rstrip()


COMMIT_HASH = get_commit_hash()
