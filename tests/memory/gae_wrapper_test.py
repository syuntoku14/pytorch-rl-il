import pytest
import random
import torch
from torch.optim import Adam
import numpy as np
import gym
import torch_testing as tt
from rlil.approximation import VNetwork, FeatureNetwork
from rlil.environments import State, Action, GymEnvironment
from rlil.memory import ExperienceReplayBuffer, GaeWrapper
from rlil.presets.continuous.models import fc_actor_critic


class DummyFeatures:
    def target(self, states):
        return states


class DummyV:
    def target(self, feature):
        return torch.ones(len(feature))


@pytest.fixture
def setUp(use_cpu):
    env = GymEnvironment('LunarLanderContinuous-v2')
    buffer = ExperienceReplayBuffer(1000, env)
    gae_buffer = GaeWrapper(buffer, discount_factor=1, lam=0.3)

    # base buffer
    states = [env.observation_space.sample() for i in range(4)]
    actions = [env.action_space.sample() for i in range(3)]
    states = State(torch.tensor(states))
    states, next_states = states[:-1], states[1:]
    actions = Action(torch.tensor(actions))
    rewards = torch.arange(0, 3, dtype=torch.float)
    gae_buffer.store(states, actions, rewards, next_states)

    feature_nw = DummyFeatures()
    v = DummyV()
    yield gae_buffer, feature_nw, v


def test_advantage(setUp):
    gae_buffer, feature_nw, v = setUp

    states, _, rewards, next_states = gae_buffer.get_all_transitions()
    values = v.target(feature_nw.target(states))
    next_values = v.target(feature_nw.target(next_states))
    advantages = gae_buffer.compute_gae(rewards, values,
                                        next_values, next_states.mask)

    # rewards: [0, 1, 2]
    # td_errors: [0, 1, 2]
    expected = torch.tensor([0 + 1 * 0.3 + 2 * 0.3 * 0.3,
                             1 + 2 * 0.3,
                             2])
    tt.assert_almost_equal(
        advantages,
        (expected - expected.mean()) / expected.std(), decimal=3)
