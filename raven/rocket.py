"""Contains the main rocket class and other vehicle specific code."""

from copy import deepcopy
import numpy as np

from .reference_frame import Frame, map_inertia_tensor
from .modules import get_properties



class Rocket():
    """This is the vehicle simulated within Raven."""

    def __init__(self, cfg:dict) -> None:

        self.frame = Frame()
        self.x = np.zeros(3, float)
        self.v = np.zeros(3, float)
        self.q = np.array([1, 0, 0, 0], float)

        self.current_stage = 0
        self.stages = []
        self.stage_roots = []

        self.throttle = 0.0
        self.active = False

        # TODO: get all available module types
        modules = {}

        for s in cfg['stages']:

            stage = {'roots':{}, 'modules':{}}

            for k, r in s['roots'].items():
                root = Frame()
                root.place_axis_angle(r['axis'], r['ang'], r['origin'])
                stage['roots'].update({k:root})

            for k, m in s['modules'].items():
                module = modules.get(m['type'])(m['args'])
                stage['modules'].update({k:module})

            self.stages.append(stage)

        # update mass, com, moi
        self.calc_inertial_properties()


    def calc_inertial_properties(self) -> None:
        """Recalculates inertial properties of rocket."""

        self.mass = 0
        x_stages = np.zeros((len(self.stages), 3), float)
        self.moi = np.zeros((3, 3), float)

        # calculate the properties of each stage
        for s, stg in enumerate(self.stages):
            m, x, i = get_properties(stg['modules'], stg['roots'])
            self.mass += m
            x_stages[s, :] = m*self.stage_roots[s].local2parent(x, True)
            self.moi += map_inertia_tensor(deepcopy(self.stage_roots[s]).invert(), m, x, i)

        self.com = np.sum(x_stages, axis=1) / self.mass


    def find_modules_with_tag(self, stage:int, tag:str) -> list:
        """Returns all modules in a stage with a given tag."""

        return [m for m in self.stages[stage].values() if tag in m.tags]


    # def set_engine_state(self) -> None:
    #     """Activates all engines on current stage."""

    #     for motor in self.find_modules_with_tag(self.current_stage, 'motor'):
    #         motor.active = True
