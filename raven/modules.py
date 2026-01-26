import numpy as np

from .referenceFrame import ReferenceFrame
from .primitives import RigidBody

class Module(RigidBody):
    """The Module class represents an object composed of multiple primitives (or other compound objects), with functions for changing its properties to give dynamic
       behaviours and act as subsystems on the rocket. These are joined together to form stages."""

    def __init__(self, primitives:dict, rootFrames:dict['ReferenceFrame']) -> None:
        
        self.primitives = primitives
        self.rootFrames = rootFrames
        self.mass = 0.0
        self.com = np.zeros(3, float)
        self.moi = np.zeros((3, 3), float)
        

    def calcInertial(self) -> tuple['float', 'np.ndarray', 'np.ndarray']:

        m = sum(p.mass for p in self.primitives)
        
        x = np.zeros(3)
        for key in self.primitives:
            x += self.primitives[key].mass*self.rootFrames[key].local2parent(self.primitives[key].com)
        
        x /= m

        I = np.zeros((3, 3), float)
        for key in self.primitives:
            I += self.primitives[key].transformInertiaTensor(frame=self.rootFrames[key], ref=self.primitives[key].ref)

        return m, x, I



class Fin(Module):

    def __init__(self, profile:str, thickness:float, **kwargs) -> None:

        # give a higher level geometry input method (profile, dims)
        pts = []

        # calculate the triangular prism mesh
        self.buildMesh(pts)

        # build and solve inertial properties of prisms

        # combine for total inertial properties

     
    def buildMesh(self, pts:list['np.ndarray']) -> tuple['list']:

        tris = [] # store triples of points by index in this list

        


        # find monotonic polygons

        # do any adjacent point lines intersect the current horizontal line 
        # as it sweeps down the shape? (if more than 2 intersections, a split is required)


        # add a meshing file with 2d (and maybe later 3d) meshing algorithms
        # split into make monotonic, choose next point, etc.

        return tuple(tris)



# class SolidMotor(Module):

#     def __init__(self, geometry:dict, fArray:list, tArray:list, isp:float, propellant:Material, wallMaterial:Material):

#         self.isp = isp
#         self.fArr = fArray
#         self.tArr = tArray

#         # build the motor casing from primitives
#         self.primitives = {}
#         self.rootFrames = {} # reference frame mapping to primitive root within module parent frame

#         odCasing = geometry['casing-diameter']
#         idCasing = odCasing - 2 * geometry['casing-thickness']
#         lNozzle = 0.5 * (geometry['exit-diameter'] - geometry['throat-diameter']) / sin(geometry['nozzle-half-angle'])
#         tProj = geometry['nozzle-thickness'] / cos(geometry['nozzle-half-angle'])
#         lGrain = geometry['casing-length'] - 2 * geometry['casing-thickness']

#         self.primitives.update({'casing':Conic(length=geometry['casing-length'], 
#                                                dOuterRoot=odCasing,
#                                                dOuterEnd=odCasing,
#                                                dInnerRoot=idCasing,
#                                                dInnerEnd=idCasing,
#                                                name='casing',
#                                                material=wallMaterial)})
        
#         self.primitives.update({'fore-bulkhead':Conic(length=geometry['casing-thickness'],
#                                                       dOuterRoot=idCasing,
#                                                       dOuterEnd=idCasing,
#                                                       dInnerRoot=0,
#                                                       dInnerEnd=0,
#                                                       name='fore-bulkhead',
#                                                       material=wallMaterial)})
        

#         self.primitives.update({'aft-bulkhead':Conic(length=geometry['casing-thickness'],
#                                                      dOuterRoot=idCasing,
#                                                      dOuterEnd=idCasing,
#                                                      dInnerRoot=geometry['throat-diameter'],
#                                                      dInnerEnd=geometry['throat-diameter'],
#                                                      name='aft-bulkhead',
#                                                      material=wallMaterial)})
        

#         self.primitives.update({'nozzle':Conic(length=lNozzle,
#                                                dOuterRoot=geometry['throat-diameter'] + tProj,
#                                                dOuterEnd=geometry['exit-diameter'] + tProj,
#                                                dInnerRoot=geometry['throat-diameter'],
#                                                dInnerEnd=geometry['exit-diameter'],
#                                                name='nozzle',
#                                                material=wallMaterial)})
        

#         self.primitives.update({'grain':Conic(length=lGrain,
#                                              dOuterRoot=idCasing,
#                                              dOuterEnd=idCasing,
#                                              dInnerRoot=geometry['fuel-port'],
#                                              dInnerEnd=geometry['fuel-port'],
#                                              name='grain',
#                                              material=propellant)})
                    

#         self.rootFrames.update({'casing':ReferenceFrame()})
#         self.rootFrames.update({'fore-bulkhead':ReferenceFrame()})
#         self.rootFrames.update({'aft-bulkhead':ReferenceFrame(origin=np.array([-geometry['casing-length'] + geometry['casing-thickness'], 0, 0], float))})
#         self.rootFrames.update({'nozzle':ReferenceFrame(origin=np.array([-geometry['casing-length'], 0, 0], float))})
#         self.rootFrames.update({'grain':ReferenceFrame(origin=np.array([-geometry['casing-thickness'], 0, 0], float))})

#         self.thrust = 0.0
#         self.onTime = 0.0
#         self.activated = False # ignites the motor, allows the timer to count up

#         self.update(0.0)


#     def update(self, dt) -> None:

#         self.onTime += dt
#         self.thrust = np.interp(self.onTime, self.tArr, self.fArr)
#         massFlow = self.thrust / (9.80665*self.isp)

#         grain = self.primitives['grain']

#         r0 = grain.rInnerRoot
#         dr = sqrt(r0**2 + massFlow*dt / (pi*grain.length*grain.material.density))
#         dNew = 2*(r0 + dr)

#         if dNew > grain.dOuterRoot:
#             dNew = grain.dOuterRoot
#             self.activated = False

#         self.primitives['grain'] = Conic(length=grain.length,
#                                          dOuterRoot=grain.dOuterRoot,
#                                          dOuterEnd=grain.dOuterEnd,
#                                          dInnerRoot=dNew,
#                                          dInnerEnd=dNew,
#                                          name='grain',
#                                          material=grain.material)

#         self.mass = self.getMass()
#         self.com = self.getCoM()
#         self.moi = self.getMoI()