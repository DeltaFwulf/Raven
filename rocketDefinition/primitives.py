from math import pi
import numpy as np
from materials import Materials



# Valid shapes:
# - ConicFull (solid frustum with limits at cone and its inverse, passing through cylinder at the midpoint)
# - ConicHollow with constant thickness wall (from cylindrical to conical limit) to be used as tubes
# - RectPrism (batteries etc)
# - fin profile (of constant thickness for the time being)
# - nosecone spline or point array



class Shape():

    shape = "default"

    def __init__(self, name="unnamed"): # TODO: use this for part indexing later, we could create a dictionary of named parts in each module?
        self.name = name

    def __str__(self):
        return f"{self.name}({self.shape})"

    def calcMass(self):
        print(f"method 'calcMass' has not been defined for shape with type {self.type}")

    def calcCoM(self): # returns the centre of mass of the part as a 3-vector relative to the part origin
        print(f"method 'calcCoM' has not been defined for shape with type {self.type}")

    def calcInertiaTensor(self): # returns the 3x3 inertia tensor of the shape about its principle axes aligned to its root coordinate system
        print(f"method 'calcMoI' has not been defined for shape with type {self.type}")



class ConicFull(Shape): # we define a full cone by its length, its top diameter and its bottom diameter

    shape = "conic_full"

    def __init__(self, dRoot, dEnd, length, name="unnamed", material="generic"):

        self.name = name

        self.rootDiameter = dRoot
        self.endDiameter = dEnd
        self.length = length

        self.rootRadius = dRoot / 2
        self.endRadius = dEnd / 2

        self.material = material

        self.mass = self.calcMass()
        self.CoM = self.calcCoM()
        self.MoI = self.calcInertiaTensor(self.CoM)


    def calcMass(self):
        return (pi * Materials.densities[self.material] * self.length / 3) * ((self.rootRadius**2) + (self.rootRadius * self.endRadius) + (self.endRadius**2))
    

    def calcCoM(self): 

        CoM = np.zeros((3), dtype=float)
        CoM[0] = (self.length / 4) * (self.rootRadius**2 + (2 * self.rootRadius * self.endRadius) + (3 * self.endRadius**2)) / (self.rootRadius**2 + (self.rootRadius * self.endRadius) + self.endRadius**2)
        return CoM
    

    # XXX: maybe we should explicitly pass in values to this method to ensure determinism?
    def calcInertiaTensor(self):

        MoI = np.zeros((3,3), dtype=float)

        if(self.rootRadius != self.endRadius):
            MoI[0,0] = (pi * Materials.densities[self.material] * self.length / 10) * (self.endRadius**5 - self.rootRadius**5) / (self.endRadius - self.rootRadius)
        
            k = (self.endRadius - self.rootRadius) / (self.length)
            xf = self.length - self.CoM
            x0 = -self.CoM

            # compacting original equation with u = (rootRadius - k*x0)
            u = self.rootRadius - (k * x0)

            # warning: whopper alert (NOTE: this has not been independently verified)
            MoI[1,1] = (pi * Materials.densities[self.material] / 2) *(\
                0.5 * (self.rootRadius - k * x0)**4 * (xf - x0) +\
                k * (self.rootRadius - k * x0)**3 * (xf**2 - x0**2) +\
                ((2 + 3*(k**2)) / 3) * (self.rootRadius - k * x0)**2 * (xf**3 - x0**3) +\
                (k * (2 + k) / 2) * (self.rootRadius - k * x0) * (xf**4 - x0**4) +\
                (k**2 * (4 + k**2) / 10) * (xf**5 - x0**5))
                
        else:
            MoI[0,0] = (pi * Materials.densities[self.material] * self.length / 2) * self.rootRadius**4

            # TODO: add cylindrical MoI here
      
        # Izz = Iyy due to axisymmetry
        MoI[2,2] = MoI[1,1]
        
        return MoI
         
