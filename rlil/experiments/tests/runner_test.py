import unittest
import numpy as np
import torch
from torch.optim import Adam
import gym
import pybullet
import pybullet_envs
import time
from rlil.environments import GymEnvironment
from rlil.utils.writer import Writer, DummyWriter
from rlil import nn
from rlil.agents import GreedyAgent
from rlil.approximation import QNetwork
from rlil.policies import SoftmaxPolicy, GaussianPolicy
from rlil.experiments import SingleEnvRunner, ParallelEnvRunner


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
        runner = SingleEnvRunner(make_discrete_agent, env, DummyWriter(), episodes=100)
        print("SingleEnv discrete exec time: {:.3f}".format(time.time() - self.startTime))
    
    def test_parallel_runner_discrete(self):
        env = gym.make('CartPole-v0')
        env = GymEnvironment(env)
        seeds = [i for i in range(5)]
        runner = ParallelEnvRunner(make_discrete_agent, env, 5, DummyWriter(), seeds, episodes=100)
        print("ParallelEnv discrete exec time: {:.3f}".format(time.time() - self.startTime))

    def test_single_runner_continuous(self):
        env = gym.make('Walker2DBulletEnv-v0')
        env = GymEnvironment(env)
        runner = SingleEnvRunner(make_continuous_agent, env, DummyWriter(), episodes=20)
        print("SingleEnv continuous exec time: {:.3f}".format(time.time() - self.startTime))
    
    def test_parallel_runner_continuous(self):
        env = gym.make('Walker2DBulletEnv-v0')
        env = GymEnvironment(env)
        seeds = [i for i in range(4)]
        runner = ParallelEnvRunner(make_continuous_agent, env, 4, DummyWriter(), seeds, episodes=20)
        print("ParallelEnv continuous exec time: {:.3f}".format(time.time() - self.startTime))