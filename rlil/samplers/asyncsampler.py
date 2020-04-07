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

    def sample(self, lazy_agent, worker_frames, worker_episodes):

        sample_info = {"frames": 0, "episodes": 0, "returns": []}

        while sample_info["frames"] < worker_frames \
                and sample_info["episodes"] < worker_episodes:
            self._env.reset()
            action = lazy_agent.act(self._env.state, self._env.reward)
            _return = 0

            while not self._env.done:
                self._env.step(action)
                sample_info["frames"] += 1
                action = lazy_agent.act(self._env.state, self._env.reward)
                _return += self._env.reward

            sample_info["episodes"] += 1
            sample_info["returns"].append(_return)

        states, actions, rewards, next_states = [], [], [], []
        for sample in lazy_agent._replay_buffer.buffer:
            state, action, reward, next_state = sample
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

        return sample_info, \
            (State.from_list(states),
             Action.from_list(actions),
             torch.tensor(rewards, dtype=torch.float),
             State.from_list(next_states)
             )


class AsyncSampler(Sampler):
    """
    AsyncSampler collects samples with asynchronous workers.
    All the workers have the same agent, which is given by the argument
    of the start_sampling method.
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
        self._work_ids = {worker: None for worker in self._workers}
        self._replay_buffer = get_replay_buffer()

    def start_sampling(self,
                       lazy_agent,
                       worker_frames=np.inf,
                       worker_episodes=np.inf):

        assert worker_frames != np.inf or worker_episodes != np.inf, \
            "worker_frames or worker_episodes must be specified"

        # start sample method if the worker is ready
        for worker in self._workers:
            if self._work_ids[worker] is None:
                self._work_ids[worker] = \
                    worker.sample.remote(
                        lazy_agent, worker_frames, worker_episodes)

    def store_samples(self, timeout=10):
        # store samples if the worker finishes sampling
        sum_sample_info = {"frames": 0, "episodes": 0, "returns": []}
        for worker, _id in self._work_ids.items():
            ready_id, remaining_id = \
                ray.wait([_id], num_returns=1, timeout=timeout)

            if len(ready_id) > 0:
                sample_info, samples = ray.get(ready_id[0])
                sum_sample_info["frames"] += sample_info["frames"]
                sum_sample_info["episodes"] += sample_info["episodes"]
                sum_sample_info["returns"] += sample_info["returns"]
                self._replay_buffer.store(*samples)
                self._work_ids[worker] = None

        return sum_sample_info
