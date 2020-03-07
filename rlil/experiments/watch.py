import os
import time
import torch
import gym
from rlil.agents import GreedyAgent
import logging


def watch(agent, env, fps=60, logger=None):
    logger = logger or logging.getLogger(__name__)
    action = None
    returns = 0
    # have to call this before initial reset for pybullet envs
    env.render(mode="human")
    while True:
        time.sleep(1 / fps)
        if env.done:
            logger.info('returns:', returns)
            env.reset()
            returns = 0
        else:
            env.step(action)
        env.render()
        action = agent.act(env.state, env.reward)
        returns += env.reward

def load_and_watch(dir, env, fps=60):
    watch(GreedyAgent.load(dir, env), env, fps=fps)