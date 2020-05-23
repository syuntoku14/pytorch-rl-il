import pytest
import torch
from torch import nn
import torch_testing as tt
from torch.optim import Adam
from rlil.environments import State, GymEnvironment
from rlil.presets.continuous.models import fc_actor_critic
from rlil.approximation import FeatureNetwork, VNetwork
from rlil.policies.gaussian import GaussianPolicy


STATE_DIM = 2


@pytest.fixture
def setUp():
    env = GymEnvironment('LunarLanderContinuous-v2')

    feature_model, value_model, policy_model = fc_actor_critic(env)
    value_optimizer = Adam(value_model.parameters())
    policy_optimizer = Adam(policy_model.parameters())
    feature_optimizer = Adam(feature_model.parameters())

    feature_nw = FeatureNetwork(feature_model, feature_optimizer)
    v = VNetwork(value_model, value_optimizer)
    policy = GaussianPolicy(policy_model, policy_optimizer, env.action_space)

    states = env.reset()
    yield states, feature_nw, v, policy


def test_share_output(setUp):
    states, feature_nw, v, policy = setUp

    states = feature_nw(states)
    value = v(states)
    action = policy(states).sample()

    value_loss = value.sum()
    policy_loss = policy(states).log_prob(action+1).sum()

    policy.reinforce(policy_loss)
    v.reinforce(value_loss)
    feature_nw.reinforce()


def test_independent_output(setUp):
    states, feature_nw, v, policy = setUp

    v_states = feature_nw(states)
    p_states = feature_nw(states)
    value = v(v_states)
    action = policy(p_states).sample()

    value_loss = value.sum()
    policy_loss = policy(p_states).log_prob(action+1).sum()

    policy.reinforce(policy_loss)
    v.reinforce(value_loss)
    feature_nw.reinforce()
