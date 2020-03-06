import numpy as np
from rlil.writer import ExperimentWriter
from .runner import SingleEnvRunner, ParallelEnvRunner

class Experiment:
    def __init__(
            self,
            agent,
            env,
            seed=0,
            n_envs=1,
            frames=np.inf,
            episodes=np.inf,
            render=False,
            quiet=False,
            write_loss=True,
    ):
        agent_name = agent.__name__
        if n_envs == 1:
            SingleEnvRunner(
                agent,
                env,
                seed=seed,
                frames=frames,
                episodes=episodes,
                render=render,
                quiet=quiet,
                writer=self._make_writer(agent_name, env.name, write_loss),
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
                writer=self._make_writer(agent_name, env.name, write_loss),
            )


    def _make_writer(self, agent_name, env_name, write_loss):
        return ExperimentWriter(agent_name, env_name, write_loss)
