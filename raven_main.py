"""Simulates the flight of a rocket on Earth."""

import numpy as np

from raven.planet import Earth
from raven.rocket import Rocket



def simulate(config:dict) -> dict:
    """Simulates a rocket's flight from launch to touchdown."""

    # tracked variables
    t = np.arange(0.0, config['tf'], config['dt'])
    x = np.zeros((1, 3), float)
    v = np.zeros((1, 3), float)
    q = np.zeros((1, 4), float)
    crds = np.zeros((1, 2), float)

    # initialise planet
    earth = Earth(x0=np.zeros(3, float), az0=0.0)

    # Rocket:
    rocket = Rocket(cfg=config['rocket'])
    x0 = earth.coords2position(lat=0, long=0, r=0, frame_out='parent')
    px = earth.pointing_vector(x=x0, pitch=0, heading=0, frame_in='parent') # FIXME: take altitude instead of r
    rocket.frame.place_base_vectors(seq=['x', 'z'],
                                    vectors=[px, x0],
                                    origin=x0)

    x[0, :] = rocket.frame.origin
    q[0, :] = rocket.frame.q
    crds[0, :] = [config['lat'], config['long']]
    # TODO: decide how to handle omega, v. (in frames or external?)

    # light engines

    # while still in flight:

        # calculate forces from current state
            # thrust force
            # aerodynamic forces
            # gravitational force
            # perturbations ?

        # update state via RK4
        # make sure to update mass etc. as well

        # perform any transformations as necessary
            # stage vehicle
            # turn off engines etc.
            # move control surfaces
            # update computer state

    # return results
