import pytest
import numpy as np
import torch
import ray
from rlil.presets.online.continuous import sac
from rlil.environments import GymEnvironment
from rlil.experiments import Experiment, Trainer
from rlil.utils.writer import Writer
from rlil.initializer import set_writer
from rlil.samplers import AsyncSampler


class MockWriter(Writer):
    def __init__(self, label):
        self.data = {}
        self.label = label
        self.sample_frames = 0
        self.sample_episodes = 1
        self.train_steps = 0

    def add_scalar(self, key, value, step="sample_frame"):
        key = key + "/" + step
        if key not in self.data:
            self.data[key] = {"values": [], "steps": []}
        self.data[key]["values"].append(value)
        self.data[key]["steps"].append(self._get_step_value(step))

    def add_text(self, name, text, step="sample_frame"):
        pass

    def _get_step_value(self, _type):
        if _type == "sample_frame":
            return self.sample_frames
        if _type == "sample_episode":
            return self.sample_episodes
        if _type == "train_step":
            return self.train_steps
        return _type


class MockExperiment(Experiment):
    def __init__(
            self,
            agent_fn,
            env,
            exp_info='default_experiments',
            num_workers=1,
            max_sample_frames=np.inf,
            max_sample_episodes=np.inf,
    ):

        # set writer
        agent_name = agent_fn.__name__
        writer = self._make_writer(agent_name, env.name, exp_info)
        set_writer(writer)

        # start training
        agent = agent_fn(env)

        sampler = AsyncSampler(env, num_workers=num_workers)
        eval_sampler = AsyncSampler(env)

        trainer = Trainer(
            agent,
            sampler,
            eval_sampler,
            max_sample_frames,
            max_sample_episodes
        )

        trainer.start_training()

    def _make_writer(self, agent_name, env_name,
                     exp_info="default_experiments"):
        self._writer = MockWriter(agent_name + '_' + env_name)
        return self._writer


def test_adds_label():
    ray.init(include_webui=False, ignore_reinit_error=True)
    env = GymEnvironment('Pendulum-v0')
    experiment = MockExperiment(sac(), env, max_sample_episodes=1)
    assert experiment._writer.label == "_sac_Pendulum-v0"


@pytest.mark.skip()
def test_writes_returns_eps():
    ray.init(include_webui=False, ignore_reinit_error=True)
    env = GymEnvironment('Pendulum-v0')
    experiment = MockExperiment(sac(), env, max_sample_episodes=3)
    np.testing.assert_equal(
        experiment._writer.data["returns/episode"]["steps"],
        np.array([1, 2, 3]),
    )


if __name__ == "__main__":
    unittest.main()
