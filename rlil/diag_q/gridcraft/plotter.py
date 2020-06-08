import itertools


import matplotlib.cm as cm
from matplotlib.patches import Rectangle, Polygon

import numpy as np
import matplotlib.pyplot as plt

NOOP = np.array([[-0.1, 0.1], [-0.1, -0.1], [0.1, -0.1], [0.1, 0.1]])
UP = np.array([[0, 0], [0.5, 0.5], [-0.5, 0.5]])
LEFT = np.array([[0, 0], [-0.5, 0.5], [-0.5, -0.5]])
RIGHT = np.array([[0, 0], [0.5, 0.5], [0.5, -0.5]])
DOWN = np.array([[0, 0], [0.5, -0.5], [-0.5, -0.5]])

TXT_OFFSET_VAL = 0.3
TXT_CENTERING = np.array([-0.08, -0.05])
TXT_NOOP = np.array([0.0, 0])+TXT_CENTERING
TXT_UP = np.array([0, TXT_OFFSET_VAL])+TXT_CENTERING
TXT_LEFT = np.array([-TXT_OFFSET_VAL, 0])+TXT_CENTERING
TXT_RIGHT = np.array([TXT_OFFSET_VAL, 0])+TXT_CENTERING
TXT_DOWN = np.array([0, -TXT_OFFSET_VAL])+TXT_CENTERING

# Action scheme
ACT_NOOP = 0
ACT_UP = 1
ACT_DOWN = 2
ACT_LEFT = 3
ACT_RIGHT = 4

ACT_OFFSETS = [
    [NOOP, TXT_NOOP],
    [UP, TXT_UP],
    [DOWN, TXT_DOWN],
    [LEFT, TXT_LEFT],
    [RIGHT, TXT_RIGHT]
]

# supports 5 actions: up, down, left, right, and noop (circle in middle)


class TabularQValuePlotter(object):
    def __init__(self, w, h, num_action=5, invert_y=True, text_values=True):
        self.w = w
        self.h = h
        self.num_action = num_action
        self.text_values = text_values
        assert num_action == 5
        self.invert_y = invert_y
        self.data = np.zeros((w, h, num_action))

    def set_value(self, x, y, action, cost):
        self.data[x, y, action] = cost

    def make_plot(self):
        plt.figure()
        ax = plt.gca()

        # scale the color since abs value of LAVA is too large
        normalized_values = np.where(self.data < 0, self.data*0.01, self.data)
        normalized_values = normalized_values - np.min(normalized_values)
        normalized_values = normalized_values/np.max(normalized_values)

        cmap = cm.get_cmap('RdYlBu')

        for x, y in itertools.product(range(self.w), range(self.h)):
            if self.invert_y:
                y = self.h-y-1
            xy = np.array([x, y])
            xy3 = np.expand_dims(xy, axis=0)

            for a in range(self.num_action-1, -1, -1):
                val = normalized_values[x, y, a]
                og_val = self.data[x, y, a]
                patch_offset, txt_offset = ACT_OFFSETS[a]
                if self.text_values:
                    xy_text = xy+txt_offset
                    ax.text(xy_text[0], xy_text[1], '%.1f' %
                            og_val, size='x-small')
                color = cmap(val)
                ax.add_patch(Polygon(xy3+patch_offset, True,
                                     color=color))
            #ax.add_patch(Rectangle(xy-0.5, 1, 1))

        ax.set_xticks(np.arange(-1, self.w+1, 1))
        ax.set_yticks(np.arange(-1, self.h+1, 1))
        plt.grid()

    def show(self):
        plt.show()


def plot_qval(gs, q_values):
    import itertools
    plotter = TabularQValuePlotter(gs.width, gs.height, text_values=True)
    for i, (x, y, a) in enumerate(itertools.product(range(gs.width), range(gs.height), range(5))):
        plotter.set_value(x, gs.height-y-1, a,
                          q_values[gs.xy_to_idx((x, y)), a])
    plotter.make_plot()
    plotter.show()


def compute_policy_deterministic(q_values, eps_greedy=0.0):
    policy_probs = np.zeros_like(q_values)
    policy_probs[np.arange(policy_probs.shape[0]), np.argmax(q_values, axis=1)] = 1.0 - eps_greedy
    policy_probs += eps_greedy / (policy_probs.shape[1])
    return policy_probs


def compute_visitation(env, policy, discount=1.0, T=50):
    dS = env.num_states
    dA = env.num_actions
    state_visitation = np.zeros((dS, 1))
    for (state, prob) in env.initial_state_distribution().items():
        state_visitation[state] = prob
    t_matrix = env.transition_matrix  # S x A x S
    sa_visit_t = np.zeros((dS, dA, T))

    norm_factor = 0.0
    for i in range(T):
        sa_visit = state_visitation * policy
        cur_discount = (discount ** i)
        sa_visit_t[:, :, i] = cur_discount * sa_visit
        norm_factor += cur_discount
        # sum-out (SA)S
        new_state_visitation = np.einsum('ij,ijk->k', sa_visit, t_matrix)
        state_visitation = np.expand_dims(new_state_visitation, axis=1)
    return np.sum(sa_visit_t, axis=2) / norm_factor
