import pytest
import random
import torch
from torch.optim import Adam
import numpy as np
import gym
import torch_testing as tt
from rlil.environments import State, Action, GymEnvironment
from rlil.memory import ExperienceReplayBuffer, AirlWrapper
from rlil.initializer import set_device
from rlil.presets.continuous.models import (fc_reward,
                                            fc_v,
                                            fc_actor_critic)
from rlil.policies import GaussianPolicy
from rlil.approximation import Approximation, FeatureNetwork, VNetwork


@pytest.fixture
def setUp(use_cpu):
    env = GymEnvironment('LunarLanderContinuous-v2')
    replay_buffer = ExperienceReplayBuffer(1000, env)

    # base buffer
    states = State(torch.tensor([env.observation_space.sample()]*100))
    actions = Action(torch.tensor([env.action_space.sample()]*99))
    rewards = torch.arange(0, 99, dtype=torch.float)
    replay_buffer.store(states[:-1], actions, rewards, states[1:])

    # expert buffer
    exp_replay_buffer = ExperienceReplayBuffer(1000, env)
    exp_states = State(torch.tensor([env.observation_space.sample()]*100))
    exp_actions = Action(torch.tensor([env.action_space.sample()]*99))
    exp_rewards = torch.arange(100, 199, dtype=torch.float)
    exp_replay_buffer.store(
        exp_states[:-1], exp_actions, exp_rewards, exp_states[1:])

    # discriminator
    reward_model = fc_reward(env)
    reward_optimizer = Adam(reward_model.parameters())
    reward_fn = Approximation(reward_model, reward_optimizer)

    value_model = fc_v(env)
    value_optimizer = Adam(value_model.parameters())
    value_fn = VNetwork(value_model, value_optimizer)

    # policy
    feature_model, _, policy_model = fc_actor_critic(env)
    feature_optimizer = Adam(feature_model.parameters())
    feature_nw = FeatureNetwork(feature_model, feature_optimizer)
    policy_optimizer = Adam(policy_model.parameters())
    policy = GaussianPolicy(policy_model, policy_optimizer, env.action_space)

    airl_buffer = AirlWrapper(replay_buffer,
                              exp_replay_buffer,
                              reward_fn,
                              value_fn,
                              policy,
                              feature_nw=feature_nw)

    samples = {
        "buffer": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
        "expert": {"states": states,
                   "actions": actions,
                   "rewards": rewards},
    }
    yield airl_buffer, samples


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


def test_store(setUp):
    gail_buffer, samples = setUp
    assert len(gail_buffer) == 99

    gail_buffer.store(samples["buffer"]["states"][:-1],
                      samples["buffer"]["actions"],
                      samples["buffer"]["rewards"],
                      samples["buffer"]["states"][1:],
                      )

    assert len(gail_buffer) == 198


def test_clear(setUp):
    gail_buffer, samples = setUp
    gail_buffer.clear()
    assert len(gail_buffer) == 0
    assert len(gail_buffer.expert_buffer) != 0
