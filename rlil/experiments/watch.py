import os
import time
import torch
import gym
from rlil.agents import GreedyAgent
from rlil.utils import get_logger
import logging
import pickle


def watch(agent, env, fps=60, dir=None):
    logger = get_logger()
    action = None
    returns = 0
    # have to call this before initial reset for pybullet envs
    env.render(mode="human")
    while True:
        time.sleep(1 / fps)
        if env.done:
            logger.info('returns: {}'.format(returns))
            env.reset()
            returns = 0
        else:
            env.step(action)
        env.render()
        action = agent.act(env.state, env.reward)
        returns += env.reward

        if len(agent.replay_buffer) % 1e4 == 0 and dir is not None:
            with open(os.path.join(dir, "buffer.pkl"), mode="wb") as f:
                pickle.dump(agent.replay_buffer.buffer, f)
                logger.info('Saved buffer. Length: {}'.format(
                    len(agent.replay_buffer)))
        if len(agent.replay_buffer) == agent.replay_buffer.capacity:
            logger.info("Buffer is full")
            break


def load_and_watch(dir, env, fps=60):
    watch(GreedyAgent.load(dir, env), env, fps=fps, dir=dir)


def load_BC_and_watch(dir, agent_fn, env, fps=60):
    watch(GreedyAgent.loadBC(dir, agent_fn, env), env, fps=fps, dir=dir)
