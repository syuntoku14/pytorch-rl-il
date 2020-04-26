import pytest
import gym
from rlil.environments import GymEnvironment
from rlil.presets.gail.continuous import gail
from rlil.presets.online.continuous import td3
from rlil.presets import validate_agent
from rlil.memory import ExperienceReplayBuffer
from rlil.environments import Action
from rlil.initializer import set_replay_buffer
from copy import deepcopy
from ..mock_agent import MockAgent


def get_transitions(env):
    replay_buffer = ExperienceReplayBuffer(1000, env)
    set_replay_buffer(replay_buffer)
    agent = MockAgent(env)

    while len(agent.replay_buffer) < 100:
        env.reset()
        while not env.done:
            env.step(agent.act(env.state, env.reward))

    return agent.replay_buffer.get_all_transitions(return_cpprb=True)


def test_bc():
    env = GymEnvironment('LunarLanderContinuous-v2')
    transitions = get_transitions(env)
    base_agent_fn = td3()
    assert len(transitions["obs"]) > 100

    validate_agent(gail(transitions=transitions,
                        base_agent_fn=base_agent_fn), env)
