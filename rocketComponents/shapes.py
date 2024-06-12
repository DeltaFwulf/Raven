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

    def calcCoM(self): # gives the CoM as a 3-vector relative to the root of the shape in vehicle coordinates (x,y,z)
        print(f"method 'calcCoM' has not been defined for shape with type {self.type}")

    def calcMoI(self, offset):
        print(f"method 'calcMoI' has not been defined for shape with type {self.type}")



class ConicFull(Shape):

    shape = "conic_full"

    def __init__(self, dRoot, dEnd, length, origin, name="unnamed", material="generic"):
        
        self.name = name
        self.origin = origin # NOTE: this origin is given relative to the root axes of the stage as a coordinate transformation [3x3]

        self.dRoot = dRoot
        self.dEnd = dEnd
        self.length = length

        self.rRoot = dRoot / 2
        self.rEnd = dEnd / 2

        self.material = material # use this to access maps in the Material class

        self.mass = self.calcMass()
        self.CoM = self.calcCoM()
        self.MoI = self.calcMoI()



    def calcMass(self):
        return (pi * Material.densities[self.material] * self.length / 3) * ((self.dRoot**2) + (self.dRoot * self.dEnd) + (self.dEnd**2))
    

    def calcCoM(self): 

        CoM = np.zeros((3), dtype=float)
        CoM[0] = (self.length / 4) * (self.rRoot**2 + (2 * self.rRoot * self.rEnd) + (3 * self.rEnd**2)) / (self.rRoot**2 + (self.rRoot * self.rEnd) + self.rEnd**2)
        return CoM
    

    # calculate the primitive's inertia tensor about its principle axes (centre of mass) such that we get all products of inertia == 0
    def calcMoI(self):

        MoI = np.zeros((3,3), dtype=float)

        # Ixx:
        if(self.dRoot == self.dEnd):
            MoI[0,0] = pi * Material.densities[self.material] * self.length * 0.5 * self.rRoot**4

        else:
            MoI[0,0] = pi * Material.densities[self.material] * self.length * 0.1 * (self.rEnd**5 - self.rRoot**5) / (self.rEnd - self.rRoot)

        # Iyy:


        # Izz:
        MoI[2,2] = MoI[1,1]

        return MoI