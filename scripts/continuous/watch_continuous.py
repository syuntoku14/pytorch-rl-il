import argparse
import pybullet
import pybullet_envs
import re
import os
import time
from rlil.environments import GymEnvironment, ENVS
from rlil.presets import continuous


def main():
    parser = argparse.ArgumentParser(description="Watch a continuous agent.")
    parser.add_argument("env", help="Name of the env")
    parser.add_argument("agent",
                        help="Name of the agent (e.g. ppo). See presets for available agents.")
    parser.add_argument("--train", action="store_true",
                        help="The model of lazy_agent: evaluation or training.")
    parser.add_argument(
        "dir", help="Directory where the agent's model was saved.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument(
        "--fps",
        default=120,
        help="Playback speed",
    )
    args = parser.parse_args()

    # load env
    env = GymEnvironment(ENVS[args.env], append_time=True)

    # load agent
    agent_fn = getattr(continuous, args.agent)()
    agent = agent_fn(env)
    agent.load(args.dir)

    # watch
    watch(agent, env, fps=args.fps, eval=not args.train)


def watch(agent, env, fps=60, eval=True):
    action = None
    returns = 0
    # have to call this before initial reset for pybullet envs
    if "Bullet" in env.name:
        env.render(mode="human")
    while True:
        time.sleep(1 / fps)
        if env.done:
            lazy_agent = agent.make_lazy_agent(evaluation=eval)
            lazy_agent.set_replay_buffer(env)
            print('returns: {}'.format(returns))
            env.reset()
            returns = 0
        else:
            env.step(action)
        env.render()
        action = lazy_agent.act(env.state, env.reward)
        returns += env.reward


if __name__ == "__main__":
    main()
