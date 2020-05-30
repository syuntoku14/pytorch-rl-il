import torch
import numpy as np
import ray
from rlil.environments import State
from rlil.initializer import is_on_policy_mode
from rlil.samplers import AsyncSampler
from rlil.experiments import Trainer


def env_validation(agent_fn, env, done_step=-1):
    """
    Args:
        agent_fn (func): presets of the agent
        env (rlil.GymEnvironment) 
        done_step (optional): 
            Run until the step reaches done_step.
            If less than 0, run until env.done == True.
    """

    agent = agent_fn(env)
    num_trains = 0

    for _ in range(2):
        env.reset()
        done_flag = False
        step = 0
        while not done_flag:
            num_trains += agent.should_train()
            if not is_on_policy_mode():
                agent.train()
            env.step(agent.act(env.state, env.reward))
            step += 1
            if done_step < 0:
                done_flag = env.done
            else:
                done_flag = done_step < step
        num_trains += agent.should_train()
        agent.train()
        agent.act(env.state, env.reward)

    assert num_trains > 0


def trainer_validation(agent_fn, env, apex=False):
    agent = agent_fn(env)
    lazy_agent = agent.make_lazy_agent()
    eval_lazy_agent = agent.make_lazy_agent(evaluation=True)
    lazy_agent.set_replay_buffer(env)
    eval_lazy_agent.set_replay_buffer(env)

    env.reset()
    action = lazy_agent.act(env.state, env.reward)

    while not env.done:
        env.step(action)
        action = lazy_agent.act(env.state, env.reward)
        _ = eval_lazy_agent.act(env.state, env.reward)

    lazy_agent.replay_buffer.on_episode_end()

    samples = lazy_agent.replay_buffer.get_all_transitions()
    samples.weights = lazy_agent.compute_priorities(samples)
    if apex:
        assert samples.weights is not None
    agent.replay_buffer.store(samples)
    agent.train()
    agent.train()
    assert agent.writer.train_steps > 1
