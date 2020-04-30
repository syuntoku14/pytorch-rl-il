import ray
import numpy as np
import os
import torch
from rlil.initializer import get_replay_buffer, call_seed
from rlil.environments import State, Action
from rlil.samplers import Sampler
from collections import defaultdict, namedtuple


StartInfo = namedtuple("StartInfo",
                       ["sample_frames",
                        "sample_episodes",
                        "train_steps"],
                       defaults=(None, ) * 3)


@ray.remote
class Worker:
    def __init__(self, make_env, seed):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self._env = make_env()
        self._env.seed(seed)

        print("Worker initialized in PID: {}".format(os.getpid()))

    def sample(self, lazy_agent, worker_frames, worker_episodes):
        """
        Args:
            lazy_agent (rlil.agent.LazyAgent): agent for sampling
            worker_frames (int): number of frames to collect
            worker_episodes (int): number of episodes to collect

        Returns:
            sample_info (StartInfo):
                keys: 
                    frames: the number of frames each episode
                    returns: the return per episode

            (States, Actions, rewards, NextStates)
        """

        sample_info = {"frames": [], "returns": []}
        lazy_agent.set_replay_buffer(self._env)

        # Sample until it reaches worker_frames or worker_episodes.
        while sum(sample_info["frames"]) < worker_frames \
                and len(sample_info["frames"]) < worker_episodes:

            self._env.reset()
            action = lazy_agent.act(self._env.state, self._env.reward)
            _return = 0
            _frames = 0

            while not self._env.done:
                self._env.step(action)
                action = lazy_agent.act(self._env.state, self._env.reward)
                _frames += 1
                _return += self._env.reward.item()

            sample_info["frames"].append(_frames)
            sample_info["returns"].append(_return)

        samples = lazy_agent._replay_buffer.get_all_transitions()

        return sample_info, samples


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
    ):
        self._env = env
        seed = call_seed()
        self._workers = [Worker.remote(env.duplicate, seed+i)
                         for i in range(num_workers)]
        self._work_ids = {worker: None for worker in self._workers}
        self._replay_buffer = get_replay_buffer()

    def start_sampling(self,
                       lazy_agent,
                       start_info=StartInfo(),
                       worker_frames=np.inf,
                       worker_episodes=np.inf,
                       ):

        # start_info has the information about when the sampling starts
        assert worker_frames != np.inf or worker_episodes != np.inf, \
            "worker_frames or worker_episodes must be specified"

        # start sample method if the worker is ready
        for worker in self._workers:
            if self._work_ids[worker] is None:
                self._work_ids[worker] = \
                    {"id": worker.sample.remote(
                        lazy_agent, worker_frames, worker_episodes),
                     "start_info": start_info}

    def store_samples(self, timeout=-1, evaluation=False):
        # if timeout < 0, wait until the sampling finishes

        # result is a dict of {start_info: {"frames": [], "returns": []}}
        result = defaultdict(lambda: {"frames": [], "returns": []})

        # store samples when the worker finishes sampling
        for worker, item in self._work_ids.items():
            _id = item["id"]
            start_info = item["start_info"]
            if timeout > 0:
                ready_id, remaining_id = \
                    ray.wait([_id], num_returns=1, timeout=timeout)
            else:
                ready_id = [_id]

            # if there is at least one finished worker
            if len(ready_id) > 0:
                # merge results
                sample_info, samples = ray.get(ready_id[0])
                result[start_info]["frames"] += sample_info["frames"]
                result[start_info]["returns"] += sample_info["returns"]

                self._work_ids[worker] = None
                if not evaluation:
                    self._replay_buffer.store(*samples)

        return result
