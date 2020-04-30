from abc import ABC, abstractmethod


class BaseReplayBuffer(ABC):
    @abstractmethod
    def store(self, states, actions, rewards, next_states):
        """Store the transition in the buffer
        Args:
            states (rlil.environment.State): batch_size x shape
            actions (rlil.environment.Action): batch_size x shape
            rewards (torch.Tensor): batch_size
            next_states (rlil.environment.State): batch_size x shape
        """

    @abstractmethod
    def sample(self, batch_size):
        '''Sample from the stored transitions'''

    @abstractmethod
    def update_priorities(self, indexes, td_errors):
        '''Update priorities based on the TD error'''

    @abstractmethod
    def get_all_transitions(self):
        '''Return all the samples'''

    @abstractmethod
    def clear(self):
        '''Clear replay buffer'''


class BaseBufferWrapper(ABC):
    def __init__(self, buffer):
        self.buffer = buffer

    def store(self, *args, **kwargs):
        self.buffer.store(*args, **kwargs)

    def sample(self, *args, **kwargs):
        return self.buffer.sample(*args, **kwargs)

    def update_priorities(self, *args, **kwargs):
        self.buffer.update_priorities(*args, **kwargs)

    def clear(self):
        self.buffer.clear()

    def get_all_transitions(self):
        return self.buffer.get_all_transitions()

    def samples_from_cpprb(self, *args, **kwargs):
        return self.buffer.samples_from_cpprb(*args, **kwargs)

    def __len__(self):
        return len(self.buffer)
