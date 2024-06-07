from math import pi
import numpy as np
from materials import Material



# Valid shapes:
# - ConicFull (solid frustum with limits at cone and its inverse, passing through cylinder at the midpoint)
# - ConicHollow with constant thickness wall (from cylindrical to conical limit) to be used as tubes
# - RectPrism (batteries etc)
# - fin profile (of constant thickness for the time being)
# - nosecone spline or point array



class Shape():

    shape = "default"

    def __init__(self, name="unnamed"):
        self.name = name

    def __str__(self):
        return f"{self.name}({self.shape})"

    def calcMass(self):
        print(f"method 'calcMass' has not been defined for shape with type {self.type}")

    def calcCoM(self): # gives the CoM as a 3-vector relative to the root of the part in vehicle coordinates (x,y,z)
        print(f"method 'calcCoM' has not been defined for shape with type {self.type}")

    def calcMoI(self, offset):
        print(f"method 'calcMoI' has not been defined for shape with type {self.type}")



class ConicFull(Shape): # we define a full cone by its length, its top diameter and its bottom diameter

    shape = "conic_full"

    def __init__(self, dTop, dBottom, length, name="unnamed", material="generic"):
        self.name = name

        self.topDiameter = dTop
        self.bottomDiameter = dBottom
        self.length = length

        self.topRadius = dTop / 2
        self.bottomRadius = dBottom / 2

        self.density = Material.density[material]


    def calcMass(self):
        return (pi * self.density * self.length / 3) * ((self.topDiameter**2) + (self.topDiameter * self.bottomDiameter) + (self.bottomDiameter**2))
    

    def calcCoM(self): 

        CoM = np.zeros((3), dtype=float)
        CoM[0] = (self.length / 4) * (self.topRadius**2 + (2 * self.topRadius * self.bottomRadius) + (3 * self.bottomRadius**2)) / (self.topRadius**2 + (self.topRadius * self.bottomRadius) + self.bottomRadius**2)
        return CoM
    

    def calcMoI(self, offset): # do we need to take into account the offset here before calculating this?

        # we are rotating this object about the rocket's centre of mass -> we need our origin point to be this location (given relative to the shape's origin at x0 or top centre)

        MoI = np.empty((3), dtype=float)

        # Jy, Jz are equal
