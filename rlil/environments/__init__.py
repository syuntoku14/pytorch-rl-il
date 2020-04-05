from .base import Environment
from .gym import GymEnvironment
from .state import State
from .action import Action, action_decorator, clip_action, squash_action

__all__ = ["Environment", "State", "GymEnvironment", "Action"]

# some example envs
# can also enter ID directly
ENVS = {
    # classic continuous environments
    "mountaincar": "MountainCarContinuous-v0",
    "lander": "LunarLanderContinuous-v2",
    # Bullet robotics environments
    "ant": "AntBulletEnv-v0",
    "cheetah": "HalfCheetahBulletEnv-v0",
    "humanoid": "HumanoidBulletEnv-v0",
    "hopper": "HopperBulletEnv-v0",
    "walker": "Walker2DBulletEnv-v0"
}