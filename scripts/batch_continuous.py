import argparse
import pybullet
import pybullet_envs
from rlil.environments import GymEnvironment, ENVS
from rlil.experiments import Experiment
from rlil.presets import batch_continuous, get_default_args
from rlil.utils import get_logger
import logging


def main():
    parser = argparse.ArgumentParser(
        description="Run a continuous actions benchmark.")
    parser.add_argument("env", help="Name of the env (see envs)")
    parser.add_argument("agent",
                        help="Name of the agent (e.g. bcq). See presets for available agents.",
                        )
    parser.add_argument("--frames", type=int, default=5e7,
                        help="The number of training frames")
    parser.add_argument("--n_envs", type=int, default=1, help="Number of envs for validation")
    parser.add_argument("--device", default="cuda",
                        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0)",
                        )
    parser.add_argument("--render", default=False,
                        help="Whether to render the environment.")
    parser.add_argument("--exp_info", default="default experiment",
                        help="One line descriptions of the experiment. Experiments' results are saved in 'runs/[exp_info]/[env_id]/'"
                        )

    args = parser.parse_args()

    if args.env in ENVS:
        env_id = ENVS[args.env]
    else:
        env_id = args.env

    env = GymEnvironment(env_id)
    agent_name = args.agent
    preset = getattr(batch_continuous, agent_name)
    preset_args = get_default_args(preset)
    agent_fn = preset(policy_path=args.policy, device=args.device)

    logger = get_logger()
    logger.setLevel(logging.DEBUG)

    args_dict = vars(args)
    args_dict.update(preset_args)

    Experiment(
        agent_fn, env, n_envs=args.n_envs, frames=args.frames, render=args.render,
        args_dict=args_dict, exp_info=args.exp_info
    )


if __name__ == "__main__":
    main()
