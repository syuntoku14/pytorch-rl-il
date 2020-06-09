from rlil.diag_q.gridcraft.grid_spec import spec_from_string
from rlil.diag_q.gridcraft.grid_env import GridEnv
from rlil.diag_q.gridcraft.wrappers import CoordinateWiseWrapper, RandomObsWrapper


class CoordinateWiseSimpleGrid(CoordinateWiseWrapper):
    def __init__(self):
        maze = spec_from_string("SOOO\\" +
                                "OOOO\\" +
                                "OOOO\\" +
                                "OORO\\")
        env = GridEnv(maze)
        super().__init__(env)


class CoordinateWiseLavaGrid(CoordinateWiseWrapper):
    def __init__(self):
        maze = spec_from_string("SOOO\\" +
                                "OLLO\\" +
                                "OOOO\\" +
                                "OLRO\\")
        env = GridEnv(maze)
        super().__init__(env)


class RandomObsSimpleGrid(RandomObsWrapper):
    def __init__(self):
        maze = spec_from_string("SOOO\\" +
                                "OOOO\\" +
                                "OOOO\\" +
                                "OORO\\")
        env = GridEnv(maze)
        super().__init__(env, 3)


class RandomObsLavaGrid(RandomObsWrapper):
    def __init__(self):
        maze = spec_from_string("SOOO\\" +
                                "OLLO\\" +
                                "OOOO\\" +
                                "OLRO\\")
        env = GridEnv(maze)
        super().__init__(env, 3)
