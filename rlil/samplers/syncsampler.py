import ray
import numpy as np
import os
import torch
from rlil.initializer import get_replay_buffer
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
        if self._env.done:
            self._env.reset()
        elif action is not None:
            self._env.step(action)

    def get_state_reward(self):
        if self._env.state is None or self._env.reward is None:
            self._env.reset()
        return (self._env.state, self._env.reward)


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
            seed=0,
    ):
        self._env = env
        self._workers = [Worker.remote(env.duplicate, seed+i)
                         for i in range(num_workers)]
        self._work_ids = []
        self._replay_buffer = get_replay_buffer()
        self._lazy_agent = None

    def start_sampling(self, lazy_agent, *args, **kwargs):
        self._lazy_agent = lazy_agent
        self._lazy_agent.reset_buffer()

        # do actions
        states = []
        rewards = []
        for worker in self._workers:
            state, reward = ray.get(worker.get_state_reward.remote())
            states.append(state)
            rewards.append(reward)
        actions = lazy_agent.act(State.from_list(states),
                                 torch.tensor(rewards, dtype=torch.float))

        # step env
        ray.get([worker.step.remote(action) for action in actions])

    def store_samples(self, *args, **kwargs):
        assert self._lazy_agent is not None

        if len(self._lazy_agent.buffer["states"]) > 0:
            self._replay_buffer.store(
                self._lazy_agent.buffer["states"][0],
                self._lazy_agent.buffer["actions"][0],
                self._lazy_agent.buffer["rewards"][0],
                self._lazy_agent.buffer["next_states"][0])
