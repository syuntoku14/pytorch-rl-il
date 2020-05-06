import pytest
import gym
from rlil.environments import GymEnvironment
from rlil.presets.continuous import airl, gail, sqil, td3, sac, ppo
from rlil.presets import validate_agent
from rlil.memory import ExperienceReplayBuffer
from rlil.environments import Action
from rlil.initializer import set_replay_buffer, get_writer
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


def test_gail():
    env = GymEnvironment("MountainCarContinuous-v0")
    transitions = get_transitions(env)
    base_agent_fn = td3(replay_start_size=0)
    assert len(transitions["obs"]) > 100

    validate_agent(gail(transitions=transitions,
                        base_agent_fn=base_agent_fn,
                        replay_start_size=10), env, done_step=50)

    writer = get_writer()
    assert writer.train_steps > 1


def test_sqil():
    env = GymEnvironment("MountainCarContinuous-v0")
    transitions = get_transitions(env)
    base_agent_fn = sac(replay_start_size=0)
    assert len(transitions["obs"]) > 100

    validate_agent(sqil(transitions=transitions,
                        base_agent_fn=base_agent_fn,
                        replay_start_size=10), env, done_step=50)

    writer = get_writer()
    assert writer.train_steps > 1


def test_airl():
    env = GymEnvironment("MountainCarContinuous-v0")
    transitions = get_transitions(env)
    base_agent_fn = ppo(replay_start_size=0)
    assert len(transitions["obs"]) > 100

    validate_agent(airl(transitions=transitions,
                        base_agent_fn=base_agent_fn,
                        replay_start_size=10), env, done_step=50)

    writer = get_writer()
    assert writer.train_steps > 1
