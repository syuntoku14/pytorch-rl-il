import pytest
import random
import torch
from torch.optim import Adam
import numpy as np
import gym
import torch_testing as tt
from rlil.environments import State, Action, GymEnvironment
from rlil.memory import ExperienceReplayBuffer, GailWrapper
from rlil.presets.gail.continuous.models import fc_discriminator
from rlil.approximation import Discriminator
from rlil.initializer import set_device


@pytest.fixture
def setUp(use_cpu):
    env = GymEnvironment('LunarLanderContinuous-v2')
    replay_buffer = ExperienceReplayBuffer(100, env)

    states = State(torch.tensor([env.observation_space.sample()]*100))
    actions = Action(torch.tensor([env.action_space.sample()]*100))
    rewards = torch.arange(0, 100, dtype=torch.float)
    replay_buffer.store(states[:-1], actions, rewards, states[1:])

    exp_replay_buffer = ExperienceReplayBuffer(100, env)
    exp_states = State(torch.tensor([env.observation_space.sample()]*100))
    exp_actions = Action(torch.tensor([env.action_space.sample()]*100))
    exp_rewards = torch.arange(100, 200, dtype=torch.float)
    exp_replay_buffer.store(
        exp_states[:-1], exp_actions, exp_rewards, exp_states[1:])

    discriminator_model = fc_discriminator(env)
    discriminator_optimizer = Adam(discriminator_model.parameters())
    discriminator = Discriminator(discriminator_model,
                                  discriminator_optimizer)

    gail_buffer = GailWrapper(replay_buffer, exp_replay_buffer, discriminator)

    samples = {
        "buffer": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
        "expert": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
    }
    yield gail_buffer, samples


def test_sample(setUp):
    gail_buffer, samples = setUp
    res_states, res_actions, res_rewards, res_next_states, _ = \
        gail_buffer.sample(4)

    # test states
    tt.assert_equal(res_states.features[0],
                    samples["buffer"]["states"].features[0])

    # test actions
    tt.assert_equal(res_actions.features[0],
                    samples["buffer"]["actions"].features[0])

    # test next_states
    tt.assert_equal(
        res_next_states.features[0], samples["buffer"]["states"].features[0])


def test_sample_both(setUp):
    gail_buffer, samples = setUp
    samples, expert_samples = gail_buffer.sample_both(4)