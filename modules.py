import numpy as np
from copy import deepcopy
from math import sqrt

from vectorUtil import ReferenceFrame
from primitives import *



class Module():

    """
    The Module class represents an object composed of multiple primitives (or other compound objects), with functions for changing its properties to give dynamic
    behaviours and act as subsystems on the rocket. These are joined together to form stages.
    
    We can parameterise these modules as well to make creation of different geometries easier i.e. parameterising a nosecone's shape can create and locate the required primitives to approximate desired geometry.
    """

    def __init__(self, primitives:dict, rootFrames:dict):
        
        self.primitives = primitives # Each primitive or compound within the module is stored in this list
        self.rootFrames = rootFrames # Stores a transform for each corresponding primitive (or compound) root from the module's root frame

        self.p2m = {}

        self.mass = self.getMass()
        self.com = self.getCoM()
        self.moi = self.getMoI()

        
    def getMass(self) -> float:
        """Calculates the total mass of the module"""
        mass = 0
        
        for key in self.primitives:
            mass += self.primitives[key].mass
        
        return mass
    

    def getCoM(self, recalcMass:bool=False) -> np.array:
        """Calculates the centre of mass position of the module relative to its root frame"""

        CoM = np.zeros(3)

        if recalcMass:
            self.mass = self.getMass()

        for key in self.primitives:
            CoM += self.primitives[key].mass * self.rootFrames[key].local2parent(self.primitives[key].com) / self.mass
       
        return CoM


    def getMoI(self) -> np.array:
        """Gets the moment of inertia tensor about the module centre of mass"""

        MoI = np.zeros((3,3), float)

        for key in self.primitives:

            pcom2mcom = deepcopy(self.rootFrames[key])
            pcom2mcom.translation += pcom2mcom.local2parent(self.primitives[key].com, incTranslation=False) - self.com # gives us (pcom - mcom)
            pcom2mcom.invert() # now becomes mcom - pcom (what we wanted)
            self.p2m.update({key:pcom2mcom.translation}) # TODO: remove this once testing is complete, or only generate in test modes, or somehow else get this
            
            # Transform this primitive's MoI by the pcom2mcom transform:
            MoI += pcom2mcom.transformInertiaTensor(self.primitives[key].moi, self.primitives[key].mass, self.primitives[key].com2ref)

        return MoI
    

    def getForce(self) -> np.array:
        return np.zeros((3), float)
    


class SolidMotor(Module):

    def __init__(self, geometry:dict, fArray:list, tArray:list, isp:float, propellant:Material, wallMaterial:Material):

        self.isp = isp
        self.fArr = fArray
        self.tArr = tArray

        # build the motor casing from primitives
        self.primitives = {}
        self.rootFrames = {} # reference frame mapping to primitive root within module parent frame

        odCasing = geometry['casing-diameter']
        idCasing = odCasing - 2 * geometry['casing-thickness']
        lNozzle = 0.5 * (geometry['exit-diameter'] - geometry['throat-diameter']) / sin(geometry['nozzle-half-angle'])
        tProj = geometry['nozzle-thickness'] / cos(geometry['nozzle-half-angle'])
        lGrain = geometry['casing-length'] - 2 * geometry['casing-thickness']

        self.primitives.update({'casing':Conic(length=geometry['casing-length'], 
                                               dOuterRoot=odCasing,
                                               dOuterEnd=odCasing,
                                               dInnerRoot=idCasing,
                                               dInnerEnd=idCasing,
                                               name='casing',
                                               material=wallMaterial)})
        
        self.primitives.update({'fore-bulkhead':Conic(length=geometry['casing-thickness'],
                                                      dOuterRoot=idCasing,
                                                      dOuterEnd=idCasing,
                                                      dInnerRoot=0,
                                                      dInnerEnd=0,
                                                      name='fore-bulkhead',
                                                      material=wallMaterial)})
        

        self.primitives.update({'aft-bulkhead':Conic(length=geometry['casing-thickness'],
                                                     dOuterRoot=idCasing,
                                                     dOuterEnd=idCasing,
                                                     dInnerRoot=geometry['throat-diameter'],
                                                     dInnerEnd=geometry['throat-diameter'],
                                                     name='aft-bulkhead',
                                                     material=wallMaterial)})
        

        self.primitives.update({'nozzle':Conic(length=lNozzle,
                                               dOuterRoot=geometry['throat-diameter'] + tProj,
                                               dOuterEnd=geometry['exit-diameter'] + tProj,
                                               dInnerRoot=geometry['throat-diameter'],
                                               dInnerEnd=geometry['exit-diameter'],
                                               name='nozzle',
                                               material=wallMaterial)})
        

        self.primitives.update({'grain':Conic(length=lGrain,
                                             dOuterRoot=idCasing,
                                             dOuterEnd=idCasing,
                                             dInnerRoot=geometry['fuel-port'],
                                             dInnerEnd=geometry['fuel-port'],
                                             name='grain',
                                             material=propellant)})
                    

        self.rootFrames.update({'casing':ReferenceFrame()})
        self.rootFrames.update({'fore-bulkhead':ReferenceFrame()})
        self.rootFrames.update({'aft-bulkhead':ReferenceFrame(translation=np.array([-geometry['casing-length'] + geometry['casing-thickness'], 0, 0], float))})
        self.rootFrames.update({'nozzle':ReferenceFrame(translation=np.array([-geometry['casing-length'], 0, 0], float))})
        self.rootFrames.update({'grain':ReferenceFrame(translation=np.array([-geometry['casing-thickness'], 0, 0], float))})

        self.thrust = 0.0
        self.onTime = 0.0
        self.activated = False # ignites the motor, allows the timer to count up

        self.update(0.0)


    def update(self, dt) -> None:

        self.onTime += dt
        self.thrust = np.interp(self.onTime, self.tArr, self.fArr)
        massFlow = self.thrust / (9.80665*self.isp)

        grain = self.primitives['grain']

        r0 = grain.rInnerRoot
        dr = sqrt(r0**2 + massFlow*dt / (pi*grain.length*grain.material.density))
        dNew = 2*(r0 + dr)

        if dNew > grain.dOuterRoot:
            dNew = grain.dOuterRoot
            self.activated = False

        self.primitives['grain'] = Conic(length=grain.length,
                                         dOuterRoot=grain.dOuterRoot,
                                         dOuterEnd=grain.dOuterEnd,
                                         dInnerRoot=dNew,
                                         dInnerEnd=dNew,
                                         name='grain',
                                         material=grain.material)

        self.mass = self.getMass()
        self.com = self.getCoM()
        self.moi = self.getMoI()