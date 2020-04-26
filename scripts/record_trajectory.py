import argparse
import pybullet
import pybullet_envs
import os
import time
import pickle
import numpy as np
import ray
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import set_replay_buffer, get_replay_buffer
from rlil.samplers import AsyncSampler
from rlil.environments import GymEnvironment
from rlil.presets.online import continuous


def main():
    parser = argparse.ArgumentParser(description="Record a trajectory of trained agent. \
        The trajectory will be stored as replay_buffer.pkl in the args.dir.")
    parser.add_argument(
        "dir", help="Directory where the agent's model is saved.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
    )
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers for training")
    parser.add_argument("--frames", type=int, default=1e6,
                        help="Number of frames to store")

    args = parser.parse_args()
    ray.init(include_webui=False, ignore_reinit_error=True)

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
    lazy_agent = agent.make_lazy_agent(evaluation=True, store_samples=True)

    # reset ExperienceReplayBuffer
    set_replay_buffer(ExperienceReplayBuffer(args.frames + 10, env))

    # set sampler
    sampler = AsyncSampler(env, num_workers=args.num_workers)

    # start recording
    replay_buffer = get_replay_buffer()

    while len(replay_buffer) < args.frames:
        sampler.start_sampling(
            lazy_agent, worker_episodes=1)

        sampler.store_samples(timeout=1)

    # save replay buffer
    filepath = os.path.join(args.dir, 'buffer.pkl')

    with open(filepath, mode='wb') as f:
        samples = replay_buffer.get_all_transitions(return_cpprb=True)
        pickle.dump(samples, f)

    print("Transitions (size: {}) is saved at {}".format(
        len(replay_buffer), filepath))


if __name__ == "__main__":
    main()
