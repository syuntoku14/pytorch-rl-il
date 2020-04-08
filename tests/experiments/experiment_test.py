import unittest
import numpy as np
import torch
import ray
from rlil.presets.continuous import sac
from rlil.environments import GymEnvironment
from rlil.experiments import Experiment
from rlil.utils.writer import Writer


class MockWriter(Writer):
    def __init__(self, label):
        self.data = {}
        self.label = label
        self.frames = 0
        self.episodes = 1

    def add_scalar(self, key, value, step="frame"):
        if key not in self.data:
            self.data[key] = {"values": [], "steps": []}
        self.data[key]["values"].append(value)
        self.data[key]["steps"].append(self._get_step(step))

    def add_loss(self, name, value, step="frame"):
        pass

    def add_schedule(self, name, value, step="frame"):
        pass

    def add_evaluation(self, name, value, step="frame"):
        self.add_scalar("evaluation/" + name, value, self._get_step(step))

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


class MockExperiment(Experiment):
    def _make_writer(self, agent_name, env_name,
                     exp_info="default_experiments"):
        self._writer = MockWriter(agent_name + '_' + env_name)
        return self._writer


class TestExperiment(unittest.TestCase):
    def setUp(self):
        ray.init(include_webui=False, ignore_reinit_error=True)
        np.random.seed(0)
        torch.manual_seed(0)
        self.env = GymEnvironment('Pendulum-v0')
        self.env.seed(0)
        self.experiment = None

    def test_adds_label(self):
        experiment = MockExperiment(sac(), self.env, max_episodes=3)
        self.assertEqual(experiment._writer.label, "_sac_Pendulum-v0")

    def test_writes_returns_eps(self):
        experiment = MockExperiment(sac(), self.env, max_episodes=3)
        np.testing.assert_equal(
            experiment._writer.data["evaluation/returns/episode"]["steps"],
            np.array([1, 2, 3]),
        )


if __name__ == "__main__":
    unittest.main()