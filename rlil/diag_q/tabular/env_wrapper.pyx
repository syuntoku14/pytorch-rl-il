# distutils: language=c++
from libcpp.map cimport map, pair

cimport tabular_env

from cython.operator cimport dereference, preincrement

cdef class TabularEnvWrapper(tabular_env.TabularEnv):
    def __init__(self, 
                 tabular_env.TabularEnv wrapped_env):
        self.wrapped_env = wrapped_env
        self.num_states = self.wrapped_env.num_states
        self.num_actions = self.wrapped_env.num_actions
        self.observation_space = self.wrapped_env.observation_space
        self.action_space = self.wrapped_env.action_space
        self.initial_state_distribution = self.wrapped_env.initial_state_distribution

    cdef map[int, double] transitions_cy(self, int state, int action):
        return self.wrapped_env.transitions_cy(state, action)

    cpdef double reward(self, int state, int action, int next_state):
        return self.wrapped_env.reward(state, action, next_state)

    cpdef observation(self, int state):
        return self.wrapped_env.observation(state)

    cpdef tabular_env.TimeStep step_state(self, int action):
        return self.wrapped_env.step_state(action)

    cpdef int reset_state(self):
        return self.wrapped_env.reset_state()

    # cpdef transition_matrix(self):
    #    return self.wrapped_env.transition_matrix()

    # cpdef reward_matrix(self):
    #    return self.wrapped_env.reward_matrix()

    cpdef set_state(self, int state):
        return self.wrapped_env.set_state(state)

    cpdef int get_state(self):
        return self.wrapped_env.get_state()

    cpdef render(self):
        return self.wrapped_env.render()

