cimport env_wrapper

cdef class TimeLimitWrapper(env_wrapper.TabularEnvWrapper):
    cdef public int num_states_origin
    cdef int _time_limit
    cdef int _timer
    cpdef (int, int) unwrap_state(self, int wrapped_state)
    cpdef int wrap_state(self, int state, int time)
