import torch
from rlil.environments import State
from rlil.experiments import ParallelEnvRunner


def validate_agent(agent_fn, env):
    validate_single_env_agent(agent_fn, env)
    validate_parallel_env_agent(agent_fn, env)


def validate_single_env_agent(agent_fn, env):
    agent = agent_fn(env)
    # Run two episodes, enough to
    # exercise all parts of the agent
    # in most cases.
    for _ in range(2):
        env.reset()
        while not env.done:
            env.step(agent.act_and_train(env.state, env.reward))
        agent.act_and_train(env.state, env.reward)


def validate_parallel_env_agent(agent_fn, env):
    ParallelEnvRunner(
        agent_fn,
        env,
        3,
        seeds=[i + 0 for i in range(3)],
        episodes=2,
    )
