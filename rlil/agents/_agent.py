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
        Update internal parameters
        """
        pass

    def act_and_train(self, state, reward):
        """
        Select an action for the current timestep and update internal parameters.

        Args:
            state (rlil.environment.State): The environment state at the current timestep.
            reward (torch.Tensor): The reward from the previous timestep.

        Returns:
            rllib.Action: The action to take at the current timestep.
        """

        self.train()
        actions = self.act(state, reward)
        return actions