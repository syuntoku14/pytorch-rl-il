from abc import ABC, abstractmethod


class Sampler(ABC):
    """
    Abstract sampler class
    """

    @abstractmethod
    def start_sampling(self, agent, max_frames, max_episodes):
        """
        Start sampling until it reaches max_frames or max_episodes.

        Args:
            agent (rlil.agent): Agent to collect samples
            max_frames (int): sampler terminates when it collects max_frames
            max_episodes (int): sampler terminates when it reaches max_episodes
        """

    @abstractmethod
    def store_samples(self):
        """
        Store collected samples to the replay_buffer
        """
