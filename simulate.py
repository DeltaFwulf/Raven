"""Simulates the flight of a rocket on Earth."""

import numpy as np

from raven.planet import Earth



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

    # initialise rocket on planet

        # turn on engines etc.

    # while not touched down

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
