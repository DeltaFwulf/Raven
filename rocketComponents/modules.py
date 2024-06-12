import numpy as np


# define all module types that can be added to the rocket

# OPEN PROBLEMS

# a module can be built from any number of primitives of different shapes and root locations, by adding them to the module before using it. Each primitive shape may be made from a specific material as well, allowing for quite detailed subassemblies to be configured


class Module():

    # location has two parts, distance from the root of the stage, and azimuth about the root azimuth of the stage
    # count refers to the number of radially symmetric copies of the module
    def __init__(self, location, count=1, name="unnamed", modType="defaultModule"):
        self.location = location
        self.mass = 0
        self.count = count
        self.name = name
        self.modType = modType
   

    # calculate and return the module's mass
    def getMass(self):
        print(f"method getMass() has not been written for module of type {self.modType}")
        return 0
    

    def getCoM(self):
        print(f"method 'getCoM' has not been defined for module of type {self.modType}")
        return np.zeros((3))


    def getMoI(self, offset):
        print(f"method 'getMoI' has not been defined for module of type {self.modType}")
