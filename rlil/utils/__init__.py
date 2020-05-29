from collections import namedtuple

Samples = namedtuple("Samples",
                     ["states", "actions", "rewards", "next_states",
                      "weights", "indexes"])
Samples.__new__.__defaults__ = Samples(*[None]*6)


def samples_to_np(samples):
    np_states, np_dones = samples.states.raw_numpy()
    np_actions = samples.actions.raw_numpy()
    np_rewards = samples.rewards.detach().cpu().numpy()
    np_next_states, np_next_dones = samples.next_states.raw_numpy()
    return np_states, np_rewards, np_actions, np_next_states, \
        np_dones, np_next_dones
