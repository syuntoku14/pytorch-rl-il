import argparse
import pybullet
import pybullet_envs
import re
import os
import time
from rlil.environments import GymEnvironment
from rlil.presets import continuous


def main():
    parser = argparse.ArgumentParser(description="Watch a continuous agent.")
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
    if args.dir[-1] != "/":
        args.dir += "/"
    env_id = args.dir.split("/")[-3]
    env = GymEnvironment(env_id)

    # load agent
    agent_name = os.path.basename(
        os.path.dirname(args.dir)).split("_")[0]
    agent_fn = getattr(continuous, agent_name)()
    agent = agent_fn(env)
    agent.load(args.dir)
    lazy_agent = agent.make_lazy_agent(evaluation=True)
    lazy_agent.set_replay_buffer(env)

    # watch
    watch(lazy_agent, env, fps=args.fps)


def watch(agent, env, fps=60):
    action = None
    returns = 0
    # have to call this before initial reset for pybullet envs
    env.render(mode="human")
    while True:
        time.sleep(1 / fps)
        if env.done:
            print('returns: {}'.format(returns))
            env.reset()
            returns = 0
        else:
            env.step(action)
        env.render()
        action = agent.act(env.state, env.reward)
        returns += env.reward


if __name__ == "__main__":
    main()
