import ray
import numpy as np
import os
import torch
from rlil.initializer import get_replay_buffer, get_seed
from rlil.environments import State, Action
from rlil.samplers import Sampler


@ray.remote
class Worker:
    def __init__(self, make_env, seed):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self._env = make_env()
        self._env.seed(seed)

        print("Worker initialized in PID: {}".format(os.getpid()))

    def step(self, action):
        self._env.step(action)

    def get_state_reward(self):
        frames = 1
        episodes = 0
        if self._env.done:
            self._env.reset()
            frames = 0
            episodes = 1
        return frames, episodes, self._env.state, self._env.reward


class SyncSampler(Sampler):
    """
    SyncSampler collects samples with synchronous workers.
    The agent collects samples similar to the VectorEnv of chainerrl.
    See Figure 1 of this paper: https://arxiv.org/pdf/1803.02811.pdf
    """

    def __init__(
            self,
            env,
            num_workers=1,
    ):
        self._env = env
        seed = get_seed()
        self._workers = [Worker.remote(env.duplicate, seed+i)
                         for i in range(num_workers)]
        self._work_ids = []
        self._replay_buffer = get_replay_buffer()
        self._lazy_agent = None

    def start_sampling(self, lazy_agent, *args, **kwargs):
        self._lazy_agent = lazy_agent

    def store_samples(self, *args, **kwargs):
        # bug here
        assert self._lazy_agent is not None
        sample_info = {"frames": 0, "episodes": 0, "returns": []}

        # do actions
        states = []
        rewards = []
        for worker in self._workers:
            frames, episodes, state, reward = ray.get(
                worker.get_state_reward.remote())
            sample_info["episodes"] += episodes
            sample_info["frames"] += frames
            states.append(state)
            rewards.append(reward)
        actions = self._lazy_agent.act(State.from_list(states),
                                       torch.tensor(rewards, dtype=torch.float))

        # step env
        ray.get([worker.step.remote(action) for action in actions])

        return sum_frames, sum_episodes
