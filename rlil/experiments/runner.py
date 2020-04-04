import logging
from rlil.environments import State
from rlil.initializer import get_logger, get_writer
import numpy as np
import torch
import warnings
import os
from abc import ABC, abstractmethod
from timeit import default_timer as timer
from torch.multiprocessing import Pipe
from torch.multiprocessing import Process
import torch.multiprocessing
torch.multiprocessing.set_start_method('spawn', True)


class EnvRunner(ABC):
    def __init__(
            self,
            agent_fn,
            env,
            seed=0,
            frames=np.inf,
            episodes=np.inf,
            render=False,
    ):
        self._agent = agent_fn(env)
        self._env = env
        self._writer = get_writer()
        self._max_frames = frames
        self._max_episodes = episodes
        self._render = render
        self._best_returns = -np.inf
        self._returns100 = []
        np.random.seed(seed)
        torch.manual_seed(seed)
        self._env.seed(seed)
        self._logger = get_logger()

        self.run()

    @abstractmethod
    def run(self):
        pass

    def _done(self):
        return (
            self._writer.frames > self._max_frames or
            self._writer.episodes > self._max_episodes
        )

    def _log(self, returns, fps):
        self._logger.info("episode: %i, frames: %i, fps: %d, returns: %d" %
                            (self._writer.episodes, self._writer.frames, fps, returns))
        if returns > self._best_returns:
            self._best_returns = returns
        self._returns100.append(returns)
        if len(self._returns100) == 100:
            mean = np.mean(self._returns100)
            std = np.std(self._returns100)
            self._writer.add_summary('returns100', mean, std, step="frame")
            self._returns100 = []
        self._writer.add_evaluation('returns/episode', returns, step="episode")
        self._writer.add_evaluation('returns/frame', returns, step="frame")
        self._writer.add_evaluation(
            "returns/max", self._best_returns, step="frame")
        self._writer.add_scalar('fps', fps, step="frame")


class SingleEnvRunner(EnvRunner):
    def run(self):
        while not self._done():
            self._run_episode()

    def _run_episode(self):
        start_time = timer()
        start_frames = self._writer.frames
        returns = self._run_until_terminal_state()
        end_time = timer()
        fps = (self._writer.frames - start_frames) / (end_time - start_time)
        self._log(returns, fps)
        self._writer.episodes += 1

    def _run_until_terminal_state(self):
        agent = self._agent
        env = self._env

        env.reset()
        returns = 0
        self._agent.train()
        action = self._agent.act(env.state, env.reward)

        while not env.done:
            self._writer.frames += 1
            if self._render:
                env.render()
            env.step(action)
            returns += env.reward
            self._agent.train()
            action = self._agent.act(env.state, env.reward)

        return returns


def worker(remote, make_env):
    env = make_env()
    print("env generated at process ID: {}".format(os.getpid()))
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                if env.done:
                    env.reset()
                elif data is not None:
                    env.step(data)  # do action
            elif cmd == 'reset':
                env.reset()
            elif cmd == 'get_state_reward_done':
                remote.send((env.state, env.reward, env.done))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.action_space, env.observation_space))
            elif cmd == 'seed':  # TODO: incorrect seeds?
                np.random.seed(data)
                torch.manual_seed(data)
                env.seed(data)
            else:
                raise NotImplementedError
    finally:
        env.close()


class ParallelEnvRunner(EnvRunner):
    def __init__(
            self,
            agent_fn,
            env,
            n_envs,
            seeds,
            **kwargs
    ):
        self._n_envs = n_envs
        self._returns = None
        self._start_time = None
        self._closed = False
        self._env = env
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.ps = \
            [Process(target=worker, args=(work_remote, env.duplicate))
             for work_remote in self.work_remotes]
        for p in self.ps:
            p.start()
        # set seeds
        for remote, seed in zip(self.remotes, seeds):
            remote.send(('seed', seed))
        super().__init__(agent_fn, env, **kwargs)

    def run(self):
        self._reset()
        while not self._done():
            self._step()
            for i, p in enumerate(self.ps):
                assert p.is_alive(), "Process {} is dead".format(i)

    def _reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(('reset', None))
        self._returns = torch.zeros(
            (self._n_envs),
            dtype=torch.float
        )
        self._start_time = timer()

    def _step(self):
        # get current state
        for remote in self.remotes:
            remote.send(('get_state_reward_done', None))
        states, rewards, dones = [], [], []
        for i, remote in enumerate(self.remotes):
            state, reward, done = remote.recv()
            states.append(state)
            rewards.append(reward)
            if done:
                self._returns[i] += reward.item()
                fps = self._writer.frames / (timer() - self._start_time)
                self._log(self._returns[i].item(), fps)
                self._returns[i] = 0
                self._writer.episodes += 1
            else:
                self._returns[i] += reward.item()
                self._writer.frames += 1

        states = State.from_list(states)
        rewards = torch.tensor(
            rewards,
            dtype=torch.float
        )
        # do actions
        self._agent.train()
        actions = self._agent.act(states, rewards)

        for i, remote in enumerate(self.remotes):
            remote.send(('step', actions[i]))

    def __del__(self):
        if not self._closed:
            self.close()

    def _assert_not_closed(self):
        assert not self._closed, "This env is already closed"

    def close(self):
        self._assert_not_closed()
        self._closed = True
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
