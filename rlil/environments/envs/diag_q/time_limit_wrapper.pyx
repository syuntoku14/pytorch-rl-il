# distutils: language=c++

from cython.operator cimport dereference, preincrement
from libcpp.map cimport map, pair
cimport env_wrapper
cimport tabular_env

cdef class TimeLimitWrapper(env_wrapper.TabularEnvWrapper):
    def __init__(self, tabular_env.TabularEnv env, int time_limit):
        super(TimeLimitWrapper, self).__init__(env)
        self.num_states_origin = self.num_states
        self.num_states = self.num_states * time_limit + 1  # 1 is done state
        self._timer = 0
        self._time_limit = time_limit
    
    cpdef int reset_state(self):
        self._timer = 0
        return self.wrapped_env.reset_state()

    @property
    def time_limit(self):
        return self._time_limit

    @property
    def timer(self):
        return self._timer

    cpdef tabular_env.TimeStep step_state(self, int action):
        ts = self.wrapped_env.step_state(action)
        self._timer += 1
        if self._timer >= self._time_limit:
            ts.done = True
        ts.state = self.wrap_state(ts.state, self._timer)
        return ts

    cdef map[int, double] transitions_cy(self, int wrapped_state, int action):
        time, state = self.unwrap_state(wrapped_state)
        transitions = self.wrapped_env.transitions_cy(state, action)
        transitions_end = transitions.end()
        transitions_it = transitions.begin()
        self._transition_map.clear()
        if time + 1 >= self._time_limit:
            self._transition_map.insert(pair[int, double](self.num_states - 1, 1.0))
            return self._transition_map

        next_time_idx = (time + 1) * self.num_states_origin
        while transitions_it != transitions_end:
            s = dereference(transitions_it).first
            p = dereference(transitions_it).second
            self._transition_map.insert(pair[int, double](s + next_time_idx, p))
            preincrement(transitions_it)
        return self._transition_map

    cpdef double reward(self, int wrapped_state, int action, int wrapped_next_state):
        timer, state = self.unwrap_state(wrapped_state)
        next_timer, next_state = self.unwrap_state(wrapped_next_state)
        if next_timer >= self._time_limit:
            return 0
        return self.wrapped_env.reward(state, action, next_state)

    cpdef observation(self, int wrapped_state):
        _, state = self.unwrap_state(wrapped_state)
        return self.wrapped_env.observation(state)

    cpdef set_state(self, int wrapped_state):
        time, state = self.unwrap_state(wrapped_state)
        self._timer = time
        return self.wrapped_env.set_state(state)
    
    cpdef (int, int) unwrap_state(self, int wrapped_state):
        # convert timestepped state to state without timestep
        time = wrapped_state // self.num_states_origin
        unwrapped_state = wrapped_state % self.num_states_origin
        return time, unwrapped_state

    cpdef (int) wrap_state(self, int state, int time):
        if time >= self.time_limit:
            return self.num_states - 1
        wrapped_state = state + self.num_states_origin * time
        return wrapped_state

    cpdef int get_state(self):
        state = self.wrapped_env.get_state()
        return self.wrap_state(state, self._timer)