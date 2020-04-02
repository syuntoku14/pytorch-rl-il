import numpy as np
import os
import torch
import signal
from abc import ABC, abstractmethod
from timeit import default_timer as timer
import torch.multiprocessing as mp
from rlil.utils import get_writer, get_logger, get_writer
from rlil.memory import get_replay_buffer
mp.set_start_method('spawn', True)


# TODO: LazyAgent class with replay_buffer

def worker(lazy_agent_class,
           shared_models,
           make_env,
           replay_dict,
           worker_id,
           seed,
           shared_frames,
           shared_episodes,
           done_event):

    # Ignore CTRL+C in the worker process
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env()
    env.seed(seed)
    lazy_agent = lazy_agent_class(shared_models)
    print("env generated at process ID: {}".format(os.getpid()))

    while not done_event.is_set():
        env.reset()
        returns = 0
        action = lazy_agent.act(env.state, env.reward)

        while not env.done:
            with shared_frames.get_lock():
                shared_frames.value += 1
            env.step(action)
            returns += env.reward
            action = lazy_agent.act(env.state, env.reward)
        with shared_episodes.get_lock():
            shared_episodes.value += 1


class ParallelEnvSampler:
    def __init__(
            self,
            lazy_agent_class,
            models,
            env,
            n_workers=1,
            seed=0,
    ):
        self._env = env
        self._done_event = mp.Event()
        self._closed = False
        self._writer = get_writer()

        # sample start
        with mp.Manager() as manager:
            self.replay_dict = manager.dict()
            self.ps = [mp.Process(target=worker,
                                  args=(lazy_agent_class,
                                        models,
                                        self._env.duplicate,
                                        self.replay_dict,
                                        worker_id,
                                        seed + worker_id,
                                        self._writer._frames,
                                        self._writer._episodes,
                                        self._done_event))
                       for worker_id in range(n_workers)]

            for p in self.ps:
                p.start()

    def sample(self):
        while not self._done():
            for i, p in enumerate(self.ps):
                assert p.is_alive(), "Process {} is dead".format(i)

    def _done(self):
        return (
            self._writer.frames > self._max_frames or
            self._writer.episodes > self._max_episodes
        )

    def __del__(self):
        if not self._closed:
            self.close()

    def _assert_not_closed(self):
        assert not self._closed, "This env is already closed"

    def close(self):
        self._assert_not_closed()
        self._closed = True
        # close processes
        self._done_event.set()
        for p in self.ps:
            p.join()