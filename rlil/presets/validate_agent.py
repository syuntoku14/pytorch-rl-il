import torch
from rlil.environments import State
from rlil.utils.writer import DummyWriter

def validate_agent(make_agent, env):
    validate_single_env_agent(make_agent, env)

def validate_single_env_agent(make_agent, env):
    agent = make_agent(env, writer=DummyWriter())
    # Run two episodes, enough to
    # exercise all parts of the agent
    # in most cases.
    for _ in range(2):
        env.reset()
        while not env.done:
            env.step(agent.act(env.state, env.reward))
        agent.act(env.state, env.reward)