class Samples:
    def __init__(self, states=None, actions=None, rewards=None,
                 next_states=None, weights=None, indexes=None):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.weights = weights
        self.indexes = indexes
        self._keys = [self.states, self.actions, self.rewards,
                      self.next_states, self.weights, self.indexes]

    def __iter__(self):
        return iter(self._keys)


def samples_to_np(samples):
    np_states, np_dones = samples.states.raw_numpy()
    np_actions = samples.actions.raw_numpy()
    np_rewards = samples.rewards.detach().cpu().numpy()
    np_next_states, np_next_dones = samples.next_states.raw_numpy()
    return np_states, np_rewards, np_actions, np_next_states, \
        np_dones, np_next_dones
