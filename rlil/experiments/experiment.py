import numpy as np
from rlil.utils.writer import ExperimentWriter
from .runner import SingleEnvRunner, ParallelEnvRunner
import os
import logging
import json


class Experiment:
    def __init__(
            self,
            agent_fn,
            env,
            args_dict={},
            exp_info="default_experiments",
            logger=None,
            seed=0,
            n_envs=1,
            frames=np.inf,
            episodes=np.inf,
            render=False,
            write_loss=True,
    ):
        agent_name = agent_fn.__name__
        writer = self._make_writer(agent_name, env.name, write_loss, exp_info)
        message = "# Parameters  \n"
        message += json.dumps(args_dict, indent=4,
                              sort_keys=True).replace("\n", "  \n")
        message += "  \n# Experiment infomation  \n" + exp_info
        writer.add_text("exp_summary", message)

        logger = logger or logging.getLogger(__name__)
        handler = logging.FileHandler(
            os.path.join(writer.log_dir, "logger.log"))
        fmt = logging.Formatter('%(levelname)s : %(asctime)s : %(message)s')
        handler.setFormatter(fmt)
        logger.addHandler(handler)

        with open(os.path.join(writer.log_dir, "args.json"), mode="w") as f:
            json.dump(args_dict, f)

        if n_envs == 1:
            SingleEnvRunner(
                agent_fn,
                env,
                seed=seed,
                frames=frames,
                episodes=episodes,
                render=render,
                writer=writer,
                logger=logger
            )
        else:
            ParallelEnvRunner(
                agent_fn,
                env,
                n_envs,
                seeds=[i + seed for i in range(n_envs)],
                frames=frames,
                episodes=episodes,
                render=render,
                writer=writer,
                logger=logger
            )

    def _make_writer(self, agent_name, env_name, write_loss, exp_info):
        return ExperimentWriter(agent_name=agent_name,
                                env_name=env_name,
                                loss=write_loss,
                                exp_info=exp_info)
