from abc import ABC, abstractmethod
from rlil.memory import ExperienceReplayBuffer
from rlil.initializer import get_n_step
from rlil.utils import Samples


class Agent(ABC):
    """
    Abstract agent class
    """

    @abstractmethod
    def act(self, state, reward=None):
        """
        Select an action for evaluation.
        If the agent has a replay-buffer, state and reward are stored.

        Args:
            state (rlil.environment.State): The environment state at the current timestep.
            reward (torch.Tensor): The reward from the previous timestep.

        Returns:
            rllib.Action: The action to take at the current timestep.
        """

    @abstractmethod
    def make_lazy_agent(self, evaluation=False):
        """
        Return a LazyAgent object for sampling or evaluation.

        Args:
            evaluation (bool, optional): If evaluation==True, the returned
            object act greedily. Defaults to False.

        Returns:
            LazyAgent: The LazyAgent object for Sampler.
        """
        pass

    def train(self):
        """
        Update internal parameters
        """
        pass

    def load(self, dirname):
        """
        Load pretrained agent.

        Args:
            dirname (str): Directory where the agent saved
        """
        pass


class LazyAgent(ABC):
    """ 
    Agent class for Sampler.
    """

    def __init__(self,
                 evaluation=False,
                 store_samples=True):
        self._states = None
        self._actions = None
        self._evaluation = evaluation
        self._store_samples = store_samples
        self.replay_buffer = None
        # for N step replay buffer
        self._n_step, self._discount_factor = get_n_step()
        if self._evaluation:
            self._n_step = 1  # disable Nstep buffer when evaluation mode

    def set_replay_buffer(self, env):
        self.replay_buffer = ExperienceReplayBuffer(
            1e7, env, n_step=self._n_step,
            discount_factor=self._discount_factor)

    def act(self, states, reward):
        """
        In the act function, the lazy_agent put a sample 
        (last_state, last_action, reward, states) into self.replay_buffer.
        Then, it outputs a corresponding action.
        """
        if self._store_samples:
            assert self.replay_buffer is not None, \
                "Call self.set_replay_buffer(env) at lazy_agent initialization."
            samples = Samples(self._states, self._actions, reward, states)
            self.replay_buffer.store(samples)
