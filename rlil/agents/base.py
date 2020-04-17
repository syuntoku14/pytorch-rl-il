from abc import ABC, abstractmethod
from rlil.utils.optim import Schedulable


class Agent(ABC, Schedulable):
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
    @abstractmethod
    def act(self, states, reward):
        """
        In the act function, the lazy_agent put a sample 
        (last_state, last_action, reward, states) into self._replay_buffer.
        Then, it outputs a corresponding action.
        """
        pass
