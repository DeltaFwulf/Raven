# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from math import sqrt, pi


"""
Created on Wed Jul 12 19:50:14 2023

@author: Michael Stoll
"""

# Rotate a point about a vector and display that in a 3D plot


def quatRotation():
    
    np.set_printoptions(suppress=True) # Suppress insignificant values for clarity
    
    vector2Rotate = np.array([0.0, 0.0, 1.0])
    
    # define a normalised rotation vector
    rotationVector = np.array([0.0, 1.0, 0.0])
    rotationVector = rotationVector / sqrt(rotationVector[0]**2 + rotationVector[1]**2 + rotationVector[2]**2)
    rotationAngleRadians = pi
    
    testQuaternion = Quaternion(axis=rotationVector, angle = rotationAngleRadians) # Rotate 0 about x=y=z
    rotatedVector = testQuaternion.rotate(vector2Rotate)
    
    # plot the original and rotated point
    print(vector2Rotate)
    print(rotatedVector)

    # plot the vectors:
    origin = np.array([0.0, 0.0, 0.0])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    

    plt.show()

    
quatRotation()