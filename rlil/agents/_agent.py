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

    def train(self):
        """
        Update internal parameters with given batch
        """
        pass


class LazyAgent(ABC):
    """ 
    Agent class for sampler.
    """
    def __init__(self, models, replay_q):
        """
        Args: 
            models (dict of torch.nn.Module): 
                memory shared torch model
            replay_q (torch.multiprocessing.Queue):
                Queue object to store samples
        """
        self.models = models
        self._replay_q = replay_q
        self._states = None
        self._actions = None
    
    @abstractmethod
    def act(self, states, reward):
        """
        In the act function, the lazy_agent put a sample of trajectories
        with (last_state, last_action, reward, states) into self._replay_q.
        Then, it outputs a corresponding action.
        """
        pass
