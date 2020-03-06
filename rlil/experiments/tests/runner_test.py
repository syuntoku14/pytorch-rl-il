import unittest
import numpy as np
import torch
from torch.optim import Adam
import gym
import pybullet
import pybullet_envs
import time
from rlil.environments import GymEnvironment
from rlil.writer import Writer
from rlil import nn
from rlil.agents import GreedyAgent
from rlil.approximation import QNetwork
from rlil.policies import SoftmaxPolicy, GaussianPolicy
from rlil.experiments import SingleEnvRunner, ParallelEnvRunner


class MockWriter(Writer):
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


def make_discrete_agent(env, writer):
    model = nn.Sequential(nn.Flatten(), nn.Linear(env.state_space.shape[0], env.action_space.n))
    optimizer = Adam(model.parameters())
    return GreedyAgent(env.action_space, q=QNetwork(model, optimizer, writer=writer))

def make_continuous_agent(env, writer):
    model = nn.Sequential(nn.Flatten(), nn.Linear(env.state_space.shape[0], env.action_space.shape[0] * 2))
    optimizer = Adam(model.parameters())
    return GreedyAgent(env.action_space, policy=GaussianPolicy(model, optimizer, env.action_space))


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def test_single_runner_discrete(self):
        env = gym.make('CartPole-v0')
        env = GymEnvironment(env)
        runner = SingleEnvRunner(make_discrete_agent, env, MockWriter(), episodes=100)
        print("SingleEnv discrete exec time: {:.3f}".format(time.time() - self.startTime))
    
    def test_parallel_runner_discrete(self):
        env = gym.make('CartPole-v0')
        env = GymEnvironment(env)
        seeds = [i for i in range(5)]
        runner = ParallelEnvRunner(make_discrete_agent, env, 5, MockWriter(), seeds, episodes=100)
        print("ParallelEnv discrete exec time: {:.3f}".format(time.time() - self.startTime))

    def test_single_runner_continuous(self):
        env = gym.make('Walker2DBulletEnv-v0')
        env = GymEnvironment(env)
        runner = SingleEnvRunner(make_continuous_agent, env, MockWriter(), episodes=20)
        print("SingleEnv continuous exec time: {:.3f}".format(time.time() - self.startTime))
    
    def test_parallel_runner_continuous(self):
        env = gym.make('Walker2DBulletEnv-v0')
        env = GymEnvironment(env)
        seeds = [i for i in range(4)]
        runner = ParallelEnvRunner(make_continuous_agent, env, 4, MockWriter(), seeds, episodes=20)
        print("ParallelEnv continuous exec time: {:.3f}".format(time.time() - self.startTime))