import numpy as np
from rlil.writer import ExperimentWriter
from .runner import SingleEnvRunner, ParallelEnvRunner
import os
import logging

class Experiment:
    def __init__(
            self,
            agent,
            env,
            logger,
            seed=0,
            n_envs=1,
            frames=np.inf,
            episodes=np.inf,
            render=False,
            quiet=False,
            write_loss=True,
    ):
        agent_name = agent.__name__
        writer = self._make_writer(agent_name, env.name, write_loss)
        get_handler = logging.FileHandler(os.path.join(writer.log_dir, "logger.log"))
        logger.addHandler(get_handler)

        if n_envs == 1:
            SingleEnvRunner(
                agent,
                env,
                seed=seed,
                frames=frames,
                episodes=episodes,
                render=render,
                quiet=quiet,
                writer=writer,
                logger=logger
            )
        else:
            ParallelEnvRunner(
                agent,
                env,
                n_envs,
                seeds=[i + seed for i in range(n_envs)],
                frames=frames,
                episodes=episodes,
                render=render,
                quiet=quiet,
                writer=writer,
                logger=logger
            )


    def _make_writer(self, agent_name, env_name, write_loss):
        return ExperimentWriter(agent_name, env_name, write_loss)
