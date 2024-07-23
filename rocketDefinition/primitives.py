from math import pi
import numpy as np
from materials import Materials



# Valid shapes:
# - ConicFull (solid frustum with limits at cone and its inverse, passing through cylinder at the midpoint)
# - ConicHollow with constant thickness wall (from cylindrical to conical limit) to be used as tubes
# - RectPrism (batteries etc)
# - fin profile (of constant thickness for the time being)
# - nosecone spline or point array


# TODO: validate or correct the inertia tensor for conic full
# TODO: except invalid values when instantiating shapes with useful error messages


class Shape():

    shape = "default"

    def __init__(self, name="unnamed"): # TODO: use this for part indexing later, we could create a dictionary of named parts in each module?
        self.name = name

    def __str__(self):
        return f"{self.name}({self.shape})"

    def calcMass(self):
        print(f"method 'calcMass' has not been defined for shape with type {self.shape}")
        return 0

    def calcCoM(self): # returns the centre of mass of the part as a 3-vector relative to the part origin
        print(f"method 'calcCoM' has not been defined for shape with type {self.shape}")
        return np.zeros((3),float)

    def calcInertiaTensor(self): # returns the 3x3 inertia tensor of the shape about its principle axes aligned to its root coordinate system
        print(f"method 'calcMoI' has not been defined for shape with type {self.shape}")
        return np.zeros((3,3),float)



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
        self.MoI = self.calcInertiaTensor()


    def calcMass(self):
        return (pi * Materials.densities[self.material] * self.length / 3) * ((self.rootRadius**2) + (self.rootRadius * self.endRadius) + (self.endRadius**2))
    

    def calcCoM(self): 

        CoM = np.zeros((3), float)
        CoM[0] = (self.length / 4) * (self.rootRadius**2 + (2 * self.rootRadius * self.endRadius) + (3 * self.endRadius**2)) / (self.rootRadius**2 + (self.rootRadius * self.endRadius) + self.endRadius**2)
        return CoM
    
    
    # TODO: catch null values
    # TODO: refactor A in terms of length, to stop having to add A and B at the final step (more integration by parts innit)
    def calcInertiaTensor(self):

        MoI = np.zeros((3,3), float)

        if(self.rootRadius != self.endRadius):
            MoI[0,0] = (pi * Materials.densities[self.material] * self.length / 10) * (self.endRadius**5 - self.rootRadius**5) / (self.endRadius - self.rootRadius)    
        else:
            MoI[0,0] = (pi * Materials.densities[self.material] * self.length / 2) * self.rootRadius**4
            
        # Iyy:
        x0 = -self.CoM[0]
        xf = self.length - self.CoM[0]

        dR = self.endRadius - self.rootRadius
        k = dR / self.length

        A = ((k**2 / 5) * (xf**5 - x0**5) + (k/2) * (self.rootRadius - k*x0) * (xf**4 - x0**4) + (1/3) * (self.rootRadius - k*x0)**2 * (xf**3 - x0**3))
        B = (self.length / 20) * ((5 * self.rootRadius**4) + (10 * dR * self.rootRadius**3) + (10 * dR**2 * self.rootRadius**2) + (5 * dR**3 * self.rootRadius) + (dR**4))

        MoI[1,1] = pi * Materials.densities[self.material] * (A + B) # this has been verified for cylindrical case as well, with errors on the order of 1e-13

        # transform to part root location (in part reference axes):
        MoI[1,1] += self.mass * self.CoM**2
    
        # Izz = Iyy due to axisymmetry
        MoI[2,2] = MoI[1,1]
        
        return MoI
    

# This shape should include the conicFull shape as a subset of the conic shape (making the shape redundant if so)
# TODO: give a simple argument to set constant thickness of the part (given the outer diameters are sufficiently large)
class Conic(Shape):

    shape = "conic"

    def __init__(self, length, dOuterRoot, dOuterEnd, dInnerRoot=0, dInnerEnd=0, name="unnamed", material="generic"):

        self.name = name
        self.material = material

        self.length = length

        self.dOuterRoot = dOuterRoot
        self.dOuterEnd = dOuterEnd
        self.dInnerRoot = dInnerRoot
        self.dInnerEnd = dInnerEnd

        self.rOuterRoot = dOuterRoot / 2
        self.rOuterEnd = dOuterEnd / 2
        self.rInnerRoot = dInnerRoot / 2
        self.rInnerEnd = dInnerEnd / 2

        self.mass = self.calcMass()
        self.CoM = self.calcCoM()
        self.MoI = self.calcInertiaTensor()

    
    def calcMass(self):
        return (pi * Materials.densities[self.material] * self.length / 3) * ((self.rOuterRoot**2 - self.rInnerRoot**2) + (self.rOuterRoot * self.rOuterEnd - self.rInnerRoot * self.rInnerEnd) + (self.rOuterEnd**2 - self.rInnerEnd**2))


    def calcCoM(self):

        CoM = np.zeros((3), float)
        CoM[0] = (self.length / 4) * ((self.rOuterRoot**2 - self.rInnerRoot**2) + 2 * (self.rOuterRoot * self.rOuterEnd - self.rInnerRoot * self.rInnerEnd) + 3 * (self.rOuterEnd**2 - self.rInnerEnd**2)) / \
                 ((self.rOuterRoot**2 - self.rInnerRoot**2) + (self.rOuterRoot * self.rOuterEnd - self.rInnerRoot * self.rInnerEnd) + (self.rOuterEnd**2 - self.rInnerEnd**2))
        
        return CoM
    

    def calcInertiaTensor(self):
        return super().calcInertiaTensor()

    


class RectangularPrism(Shape):

    shape = "rectangular_prism"

    def __init__(self, x, y, z, name="unnamed", material="generic"):

        self.name = name

        self.x = x
        self.y = y
        self.z = z

        self.material = material


    def calcMass(self):
        return Materials.densities[self.material] * self.x * self.y * self.z
    

    def calcCoM(self):
        
        CoM = np.zeros((3), float)
        CoM[0] = self.x / 2
        return CoM
    

    def calcMoI(self):

        MoI = np.zeros((3,3), float)

        #TODO: input inertia tensor calculations

        return MoI
         


def shapeTester():

    #testPrimitive = ConicFull(dRoot=0, dEnd=1, length=1, name="test_primitive", material="generic")
    testPrimitive = Conic(length=1, dOuterRoot=0.2, dOuterEnd=1, dInnerRoot=0, dInnerEnd=0.8, name="test_conic", material="generic")

    # get some information about this primitive
    print(f"the mass of {testPrimitive.name} is {testPrimitive.mass} kg")
    print(f"the centre of mass of {testPrimitive.name} is\n{testPrimitive.CoM}")
    print(f"the inertia tensor for {testPrimitive.name} is:\n{testPrimitive.MoI}")



shapeTester()