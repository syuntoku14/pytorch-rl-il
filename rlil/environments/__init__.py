from .base import Environment
from .gym import GymEnvironment
from .state import State
from .action import Action, action_decorator, clip_action, squash_action
from .reward_fns import *
import gym
from gym.envs.registration import registry, make, spec


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


# Half gravity envs
register(id='HalfGravityWalker2DBulletEnv-v0',
         entry_point='rlil.environments.rlil_envs:HalfGravityWalker2DBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)
register(id='HalfGravityHalfCheetahBulletEnv-v0',
         entry_point='rlil.environments.rlil_envs:HalfGravityHalfCheetahBulletEnv',
         max_episode_steps=1000,
         reward_threshold=3000.0)

register(id='HalfGravityAntBulletEnv-v0',
         entry_point='rlil.environments.rlil_envs:HalfGravityAntBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='HalfGravityHopperBulletEnv-v0',
         entry_point='rlil.environments.rlil_envs:HalfGravityHopperBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='HalfGravityHumanoidBulletEnv-v0',
         entry_point='rlil.environments.rlil_envs:HalfGravityHumanoidBulletEnv',
         max_episode_steps=1000)

# Double gravity envs
register(id='DoubleGravityWalker2DBulletEnv-v0',
         entry_point='rlil.environments.rlil_envs:DoubleGravityWalker2DBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)
register(id='DoubleGravityDoubleCheetahBulletEnv-v0',
         entry_point='rlil.environments.rlil_envs:DoubleGravityDoubleCheetahBulletEnv',
         max_episode_steps=1000,
         reward_threshold=3000.0)

register(id='DoubleGravityAntBulletEnv-v0',
         entry_point='rlil.environments.rlil_envs:DoubleGravityAntBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='DoubleGravityHopperBulletEnv-v0',
         entry_point='rlil.environments.rlil_envs:DoubleGravityHopperBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)

register(id='DoubleGravityHumanoidBulletEnv-v0',
         entry_point='rlil.environments.rlil_envs:DoubleGravityHumanoidBulletEnv',
         max_episode_steps=1000)

# Different gait bullet envs
register(id='HalfFrontLegsAntBulletEnv-v0',
         entry_point='rlil.environments.rlil_envs:HalfFrontLegsAntBulletEnv',
         max_episode_steps=1000,
         reward_threshold=2500.0)


__all__ = ["Environment", "State", "GymEnvironment", "Action"]

# some example envs
# can also enter ID directly
ENVS = {
    # classic continuous environments
    "pendulum": "Pendulum-v0",
    "mountaincar": "MountainCarContinuous-v0",
    "lander": "LunarLanderContinuous-v2",
    # Bullet robotics environments
    "ant": "AntBulletEnv-v0",
    "cheetah": "HalfCheetahBulletEnv-v0",
    "humanoid": "HumanoidBulletEnv-v0",
    "hopper": "HopperBulletEnv-v0",
    "walker": "Walker2DBulletEnv-v0",
    # Half gravity bullet envs
    "half_gravity_ant": "HalfGravityAntBulletEnv-v0",
    "half_gravity_cheetah": "HalfGravityHalfCheetahBulletEnv-v0",
    "half_gravity_humanoid": "HalfGravityHumanoidBulletEnv-v0",
    "half_gravity_hopper": "HalfGravityHopperBulletEnv-v0",
    "half_gravity_walker": "HalfGravityWalker2DBulletEnv-v0",
    # Double gravity bullet envs
    "double_gravity_ant": "DoubleGravityAntBulletEnv-v0",
    "double_gravity_cheetah": "DoubleGravityHalfCheetahBulletEnv-v0",
    "double_gravity_humanoid": "DoubleGravityHumanoidBulletEnv-v0",
    "double_gravity_hopper": "DoubleGravityHopperBulletEnv-v0",
    "double_gravity_walker": "DoubleGravityWalker2DBulletEnv-v0",
    # Different gait bullet envs
    "half_front_legs_ant": 'HalfFrontLegsAntBulletEnv-v0'
}


REWARDS = {
    "Pendulum-v0": PendulumReward,
    "MountainCarContinuous-v0": MountainCarContinuousReward,
}