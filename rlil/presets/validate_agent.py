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


def trainer_validation(agent_fn, env, num_workers=3):
    agent = agent_fn(env)
    ray.init(include_webui=False, ignore_reinit_error=True)
    sampler = AsyncSampler(
        env,
        num_workers=num_workers,
    ) if num_workers > 0 else None
    trainer = Trainer(agent, sampler, max_train_steps=1)
    trainer.start_training()
