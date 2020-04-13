from abc import ABC, abstractmethod


class Sampler(ABC):
    """
    Abstract sampler class
    """

    @abstractmethod
    def start_sampling(self, agent, worker_frames, worker_episodes):
        """
        Start sampling until it reaches worker_frames or worker_episodes.

        Args:
            agent (rlil.agent): Agent to collect samples
            worker_frames (int): worker stops to sample when it collects worker_frames
            worker_episodes (int): worker stops to sample when it reaches worker_episodes
        """

    @abstractmethod
    def store_samples(self):
        """
        Store collected samples to the replay_buffer

        Returns:
            result (dict): Information of sampling (e.g. stored frames, returns, etc)
        """
