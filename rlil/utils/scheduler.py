from rlil.initializer import get_writer


class LinearScheduler:
    def __init__(
            self,
            initial_value,
            final_value,
            decay_end,
            name='variable',
    ):
        self._initial_value = initial_value
        self._final_value = final_value
        self._decay_end = decay_end
        self._i = 0
        self._name = name
        self._writer = get_writer()

    def _get_value(self):
        if self._i >= self._decay_end:
            return self._final_value
        alpha = self._i / self._decay_end
        return alpha * self._final_value + (1 - alpha) * self._initial_value

    def get(self):
        result = self._get_value()
        self._writer.add_scalar("schedule/" + self._name, result)
        return result

    def update(self):
        self._i += 1
