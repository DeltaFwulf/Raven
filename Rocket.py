
# Units are SI unless specified otherwise!
# for component placement on the vehicle, x0 is the tip of the nosecone, with +x towards the motor
# for vehicle position, use standard aircraft conventions (+x towards nose)
# the default angle unit here is radians, please specify degrees if used in variable name

# Constraints:


import numpy as np
from pyquaternion import Quaternion


class Rocket:

    # add in all subsystems here
    def __init__(self):

        # initial orientation of the rocket
        self.launchRailElevation = 0 # 0 to pi/2
        self.launchRailAzimuth = 0 # 0 to 2 * pi

        """Configuration"""
        self.finCount = None
        self.bodyTubeOD = 0
        self.bodyTubeThickness = 0

        """Inertial Data"""
        self.mass = 0
        self.Ixx = 0
        self.Iyy = 0
        self.Ixx = 0
        self.CoG = 0
        self.Cp = 0
        self.position = np.array([0, 0, 0])# an X,Y,Z vector representing the position of the rocket's centre of mass
        self.orientation # a quaternion that maps the world frame UP vector (0, 0, 1) to the rocket NOSE vector

        """Propulsion"""
        self.motor = None # This will be another class TODO: find out how to do this

        """Aerodynamics"""
        self.fin = None # fin class passed into here
        self.nosecone = None
        self.boattail = None

        """Subsystem Positions"""
        self.xMotor = 0
        self.xFin = 0
        self.xRecovery = 0
        self.xAvBay = 0
        self.xPayload = 0


    def setMass(self): # sum all component masses

        return (self.fin.mass * self.finCount) + 


    # Rotate the vehicle about an axis (in world coordinates)
    def quatRotate(initOrientation, rotationAxis, rotationAngle): # rotation axis is the world-frame rocket axis
        
        rotQuaternion = Quaternion(axis = rotationAxis, angle = rotationAngle)
        newOrientation = rotQuaternion.rotate(initOrientation)

        return newOrientation



    