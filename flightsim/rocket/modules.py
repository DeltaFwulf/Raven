import numpy as np
from copy import deepcopy

from motion.vectorUtil import Transform
from rocket.primitives import *
from ui.textUtil import arrFormat
# define all module types that can be added to the rocket

# OPEN PROBLEMS

# a module can be built from any number of primitives of different shapes and root locations, by adding them to the module before using it. Each primitive shape may be made from a specific material as well, allowing for quite detailed subassemblies to be configured


class Module():

    """
    The Module class represents an object composed of multiple primitives (or other compound objects), with functions for changing its properties to give dynamic
    behaviours and act as subsystems on the rocket. These are joined together to form stages.
    
    We can parameterise these modules as well to make creation of different geometries easier i.e. parameterising a nosecone's shape can create and locate the required primitives to approximate desired geometry.
    """

    def __init__(self, primitives:list[Primitive]=[], rootTransforms:list[Transform]=[]):
        self.primitives = primitives # Each primitive or compound within the module is stored in this list
        self.rootTransforms = rootTransforms # Stores a transform for each corresponding primitive (or compound) root from the module's root frame

        self.mass = self.getMass()
        self.com = self.getCoM()
        self.moi = self.getMoI()

    # calculate and return the module's mass
    def getMass(self) -> float:
        """Calculates the total mass of the module"""
        mass = 0
        for primitive in self.primitives: mass += primitive.mass

        return mass
    

    def getCoM(self) -> np.array:
        """Calculates the centre of mass position of the module relative to its root frame"""

        CoM = np.zeros(3)
       
        for i in range(0, len(self.primitives)):
            CoM += self.primitives[i].mass * self.rootTransforms[i].local2parent(self.primitives[i].com) / self.mass

        return CoM


    def getMoI(self) -> np.array:
        """Gets the moment of inertia tensor about the module centre of mass"""

        MoI = np.zeros((3,3), float)

        for i in range(0, len(self.primitives)):

            # We have the MoI of each primitive about its own CoM, we now need it about the CoM of the module.
            pcom2mcom = deepcopy(self.rootTransforms[i])
            pcom2mcom.chain(self.primitives[i].root2com) 
            pcom2mcom.invert()
            pcom2mcom.move(translation=self.com, reference='parent')

            # Transform this primitive's MoI by the pcom2mcom transform:
            MoI += pcom2mcom.transformInertiaTensor(self.primitives[i].MoI, self.primitives[i].mass, self.primitives[i].com2ref)

        return MoI
    


class Tank(Module):

    """The Tank object represents a propellant tank to be placed on the vehicle. It has a specified propellant, volume, pressure, and proportion filled
    
    The tank is made from a specific material with a constant wall thickness. The tank is currently approximated as a cylinder for ease of calculation, however round caps are planned.

    - The tank can be automatically designed to hold hoop stress (we assume this is the maximum stress for now) by changing the wall thickness.
    - As the tank drains, its pressure and propellant mass decrease (pressure can be replenished when plumbing is introduced)
    - The CoM ofthe tank assumes that propellant has settled, though ullage issues can be simulated in the future
    - All plumbing, etc should be included either on the motor side, or in a feed object that contains all relevant pressure drops, etc.
    """

    def __init__(self):

        # tank geometry (len, width, volume)

        # tank wall material

        # tank pressure and therefore tank wall thickness

        # tank propellant type

        pass