import numpy as np
from . import rendering
import pyglet


class PendulumRender:
    def __init__(self):
        # add common image
        self.viewer = rendering.Viewer(500, 500)
        self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
        fname = "clockwise.png"
        self.img = rendering.Image(fname, 1., 1.)
        self.imgtrans = rendering.Transform()
        self.img.add_attr(self.imgtrans)
        axle = rendering.make_circle(.05)
        axle.set_color(0, 0, 0)
        self.viewer.add_geom(axle)

        self.max_torque = 2.
        self.pendulum_dict = {}

    def add_pendulum(self, name, color=(.8, .3, .3, 0.5)):
        # add pendulum
        rod = rendering.make_capsule(1.0, 0.2)
        rod.set_color(*color)
        pole_transform = rendering.Transform()
        rod.add_attr(pole_transform)
        self.viewer.add_geom(rod)

        # add legend
        legend = rendering.make_circle(0.1)
        legend.set_color(*color[:3], 1.0)
        x = 0.7
        y = 2 - len(self.pendulum_dict) * 0.3
        legend_trans = rendering.Transform(translation=(x, y))
        legend.add_attr(legend_trans)
        self.viewer.add_geom(legend)
        label = pyglet.text.Label(name, font_size=8,
                                  x=0, y=0, anchor_x='left', anchor_y='center',
                                  color=(0, 0, 0, 255))
        label = rendering.Text(label)
        label_trans = rendering.Transform(translation=(x + 0.2, y),
                                          scale=(0.015, 0.015))
        label.add_attr(label_trans)
        self.viewer.add_geom(label)
        self.pendulum_dict[name] = pole_transform

    def render(self, name, state, last_u=None, mode='human'):
        self.viewer.add_onetime(self.img)
        self.pendulum_dict[name].set_rotation(state[0] + np.pi/2)
        if last_u is not None:
            last_u = last_u
            last_u = np.clip(last_u, -self.max_torque, self.max_torque)
            self.imgtrans.scale = (-last_u/2, np.abs(last_u)/2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
