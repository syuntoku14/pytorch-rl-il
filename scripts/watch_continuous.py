# pylint: disable=unused-import
import argparse
import pybullet
import pybullet_envs
import re
import os
from rlil.environments import GymEnvironment
from rlil.experiments import GreedyAgent, watch
from rlil.presets import continuous
from continuous import ENVS
import logging
logging.basicConfig(level=logging.DEBUG)


def watch_continuous():
    parser = argparse.ArgumentParser(description="Watch a continuous agent.")
    parser.add_argument("env", help="ID of the Environment")
    parser.add_argument("dir", help="Directory where the agent's model was saved.")
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
    parser.add_argument("--use_BC", action="store_true")
    parser.add_argument("--save_buffer", action="store_true")
    args = parser.parse_args()

    if args.env in ENVS:
        env_id = ENVS[args.env]
    else:
        env_id = args.env

    env = GymEnvironment(env_id)
    if args.use_BC:
        p = re.compile("_(.*)__")
        agent_name = p.search(os.path.basename(os.path.dirname(args.dir))).group(1)
        agent_fn = getattr(continuous, agent_name)(device=args.device)
        agent = GreedyAgent.load_BC(args.dir, agent_fn, env, device=args.device)
    else:
        agent = GreedyAgent.load(args.dir, env, device=args.device)
    
    if args.save_buffer:
        watch(agent, env, fps=args.fps, dir=args.dir)
    else:
        watch(agent, env, fps=args.fps)

if __name__ == "__main__":
    watch_continuous()
