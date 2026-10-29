"""Planet definitions and helper functions."""

from copy import deepcopy
from math import pi
import numpy as np
from numpy.linalg import norm

from .vectors import unit, rotate_axis_angle
from .reference_frame import Frame

class Earth():
    """Represents the planet Earth.

    Stores the planet's state and contains planet methods used by the simulation."""

    def __init__(self, x0, az0):
        self.mu = 3.986e14
        self.r = 6378e3
        self.v = np.zeros(3, float)
        self.omega = 7.292115024e-5
        self.tilt = 0.4090926295

        self.pci = Frame()
        self.pci.place_axis_angle(axis=np.array([0,0,1], float), ang=0.0, origin=x0)
        self.pci.move(axis=np.array([1,0,0], float), ang=self.tilt, move_frame='local')

        self.pcf = deepcopy(self.pci)
        self.pcf.move(axis=np.array([0,0,1], float), ang=az0, move_frame='local')


    def convert_frame(self, frame_in:str, frame_out:str, vec:np.ndarray, inc_translation:bool=True) -> np.ndarray:
        """Maps the input vector into the desired planet vector."""

        if frame_in == frame_out:
            return vec

        # convert to parent frame
        match frame_in:
            case 'pcf':
                vec = self.pcf.local2parent(vec, inc_translation)
            case 'pci':
                vec = self.pci.local2parent(vec, inc_translation)

        match frame_out:
            case 'pcf':
                vec = self.pcf.parent2local(vec, inc_translation)
            case 'pci':
                vec = self.pci.parent2local(vec, inc_translation)

        return vec


    def gravity(self, x:np.array, frame_in:str='parent', frame_out:str='parent') -> np.array:
        """Calculates the gravitational acceleration vector given a position relative to the planet.
           Inputs and outputs can be specified in one of three frames: parent, pcr, or pcnr."""

        xloc = self.convert_frame(frame_in, 'pcf', x, inc_translation=False)
        return self.convert_frame('pcf', frame_out, self.mu*xloc / norm(xloc)**3, inc_translation=False)


    def get_altitude(self, x:np.array, frame_in:str='parent') -> float:
        """Returns altitude above mean sea level"""

        return norm(self.convert_frame(frame_in, 'pcf', x, inc_translation=False)) - self.r


    # def position2coords(self, x:np.array, frame_in:str='parent') -> tuple['float', 'float']:
    #     """Given a relative location to the planet, returns PCR latitude and longitude"""

    #     return cartesian2coords(self.convert_frame(frame_in, 'pcf', x, inc_translation=True))


    # def coords2position(self, lat:float, long:float, r:float, frame_out:str='parent') -> np.array:
    #     """Returns the position vector that gives an object the specified coordinates on the planet"""

    #     x = coords2cartesian(lat, long, r)
    #     return self.convert_frame('pcf', frame_out, x, inc_translation=frame_out == 'parent')


    def pointing_vector(self, x:np.array, pitch:float, heading:float, frame_in:str='parent') -> np.array:
        """Returns a unit vector with desired pitch and compass heading"""

        if frame_in == 'parent':
            xloc = self.pcf.parent2local(x, incTranslation=True)
        elif frame_in in ('pci', 'pcf'):
            xloc = x
        else:
            raise ValueError("frame_in must be either parent, pci, or pcf.")

        up = xloc
        east = norm(up, np.array([0,0,1], float))
        p = rotate_axis_angle(east, up, (pi / 2 - heading) % (2*pi)) # apply heading
        a = np.cross(p, up) # pitch rotation axis
        p = unit(rotate_axis_angle(p, a, pitch)) # apply pitch

        return p
