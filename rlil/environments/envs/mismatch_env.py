import os
import pybullet
import pybullet_data
from pybullet_envs.scene_stadium import SinglePlayerStadiumScene
from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
from pybullet_envs.robot_bases import MJCFBasedRobot
from robot_locomotors import (WalkerBase,
                              Hopper,
                              Walker2D,
                              HalfCheetah,
                              Ant,
                              Humanoid)


# Half gravity environments
class HalfGravityWalkerBulletEnv(WalkerBaseBulletEnv):
    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                      gravity=9.8 * 0.5,
                                                      timestep=0.0165 / 4,
                                                      frame_skip=4)
        return self.stadium_scene


class HalfGravityHopperBulletEnv(HalfGravityWalkerBulletEnv):

    def __init__(self, render=False):
        self.robot = Hopper()
        self.robot.power = 0
        HalfGravityWalkerBulletEnv.__init__(self, self.robot, render)


class HalfGravityWalker2DBulletEnv(HalfGravityWalkerBulletEnv):

    def __init__(self, render=False):
        self.robot = Walker2D()
        HalfGravityWalkerBulletEnv.__init__(self, self.robot, render)


class HalfGravityHalfCheetahBulletEnv(HalfGravityWalkerBulletEnv):

    def __init__(self, render=False):
        self.robot = HalfCheetah()
        HalfGravityWalkerBulletEnv.__init__(self, self.robot, render)

    def _isDone(self):
        return False


class HalfGravityAntBulletEnv(HalfGravityWalkerBulletEnv):
    def __init__(self, render=False):
        self.robot = Ant()
        HalfGravityWalkerBulletEnv.__init__(self, self.robot, render)


class HalfGravityHumanoidBulletEnv(HalfGravityWalkerBulletEnv):
    def __init__(self, robot=Humanoid(), render=False):
        self.robot = robot
        HalfGravityWalkerBulletEnv.__init__(self, self.robot, render)
        self.electricity_cost = \
            4.25 * HalfGravityWalkerBulletEnv.electricity_cost
        self.stall_torque_cost = \
            4.25 * HalfGravityWalkerBulletEnv.stall_torque_cost


# Double gravity environments
class DoubleGravityWalkerBulletEnv(WalkerBaseBulletEnv):
    def create_single_player_scene(self, bullet_client):
        self.stadium_scene = SinglePlayerStadiumScene(bullet_client,
                                                      gravity=9.8 * 2.0,
                                                      timestep=0.0165 / 4,
                                                      frame_skip=4)
        return self.stadium_scene


class DoubleGravityHopperBulletEnv(DoubleGravityWalkerBulletEnv):

    def __init__(self, render=False):
        self.robot = Hopper()
        self.robot.power = 0
        DoubleGravityWalkerBulletEnv.__init__(self, self.robot, render)


class DoubleGravityWalker2DBulletEnv(DoubleGravityWalkerBulletEnv):

    def __init__(self, render=False):
        self.robot = Walker2D()
        DoubleGravityWalkerBulletEnv.__init__(self, self.robot, render)


class DoubleGravityHalfCheetahBulletEnv(DoubleGravityWalkerBulletEnv):

    def __init__(self, render=False):
        self.robot = HalfCheetah()
        DoubleGravityWalkerBulletEnv.__init__(self, self.robot, render)

    def _isDone(self):
        return False


class DoubleGravityAntBulletEnv(DoubleGravityWalkerBulletEnv):
    def __init__(self, render=False):
        self.robot = Ant()
        DoubleGravityWalkerBulletEnv.__init__(self, self.robot, render)


class DoubleGravityHumanoidBulletEnv(DoubleGravityWalkerBulletEnv):
    def __init__(self, robot=Humanoid(), render=False):
        self.robot = robot
        DoubleGravityWalkerBulletEnv.__init__(self, self.robot, render)
        self.electricity_cost = \
            4.25 * DoubleGravityWalkerBulletEnv.electricity_cost
        self.stall_torque_cost = \
            4.25 * DoubleGravityWalkerBulletEnv.stall_torque_cost


# Different gait bullet envs
class HalfFrontLegsAntBulletEnv(WalkerBaseBulletEnv):
    def __init__(self, render=False):
        self.robot = Ant()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)


class RlilMJCFBasedRobot(MJCFBasedRobot):
    def reset(self, bullet_client):
        self._p = bullet_client
        # print("Created bullet_client with id=", self._p._client)
        if (self.doneLoading == 0):
            self.ordered_joints = []
            self.doneLoading = 1
            if self.self_collision:
                self.objects = self._p.loadMJCF(
                    os.path.join(os.path.dirname(__file__),
                                 "data", self.model_xml),
                    flags=pybullet.URDF_USE_SELF_COLLISION |
                    pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = \
                    self.addToScene(self._p, self.objects)
            else:
                self.objects = self._p.loadMJCF(
                    os.path.join(os.path.dirname(__file__),
                                 "data", self.model_xml))
                self.parts, self.jdict, self.ordered_joints, self.robot_body = \
                    self.addToScene(self._p, self.objects)
        self.robot_specific_reset(self._p)

        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return s


class RlilWalkerBase(RlilMJCFBasedRobot, WalkerBase):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        RlilMJCFBasedRobot.__init__(self, fn, robot_name, action_dim, obs_dim)
        self.power = power
        self.camera_x = 0
        self.start_pos_x, self.start_pos_y, self.start_pos_z = 0, 0, 0
        self.walk_target_x = 1e3  # kilometer away
        self.walk_target_y = 0
        self.body_xyz = [0, 0, 0]


class HalfFrontLegsAnt(RlilWalkerBase, Ant):
    foot_list = ['front_left_foot', 'front_right_foot',
                 'left_back_foot', 'right_back_foot']

    def __init__(self):
        RlilWalkerBase.__init__(
            self, "ant_half_front_legs.xml", "torso", action_dim=8, obs_dim=28, power=2.5)


class HalfFrontLegsAntBulletEnv(WalkerBaseBulletEnv):
    def __init__(self, render=False):
        self.robot = HalfFrontLegsAnt()
        WalkerBaseBulletEnv.__init__(self, self.robot, render)
