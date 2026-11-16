"""Contains module species and module-specific functions."""

import numpy as np

from .reference_frame import Frame
from .primitives import Conic



def get_properties(primitives:list, roots:list) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns rigid body properties of the module."""

    m = sum(p.mass for p in primitives)
    x = np.zeros(3, float)
    moi = np.zeros((3, 3), float)

    for k, p in primitives.items():
        x += p.mass*roots[k].local2parent(p.com)
        moi += p.transformInertiaTensor(frame=roots[k], ref=p.ref)

    x /= m

    return m, x, moi



class Fin():
    """Represents an aerodynamic fin.

    These can take different shapes, by a profile string and parameters."""

    def __init__(self, profile:str, thickness:float, **kwargs) -> None:

        # give a higher level geometry input method (profile, dims)
        pts = []

        # calculate the triangular prism mesh
        self.solve_mesh(pts)

        # build and solve inertial properties of prisms

        # combine for total inertial properties


    def solve_mesh(self, pts:list[np.ndarray]) -> tuple['list']:
        """Calculates the fin's inertial properties by solving a triangular mesh."""

        tris = [] # store triples of points by index in this list

        # find monotonic polygons

        # do any adjacent point lines intersect the current horizontal line 
        # as it sweeps down the shape? (if more than 2 intersections, a split is required)

        # add a meshing file with 2d (and maybe later 3d) meshing algorithms
        # split into make monotonic, choose next point, etc.

        return tuple(tris)



class Tube():
    """A hollow, cylindrical tube of constant thickness."""

    def __init__(self, diameter:float, thickness:float, length:float, density:float) -> None:

        self.tags = ('aero',)

        d_inner = diameter - 2*thickness
        tube = Conic(density=density,
                     length=length,
                     dOuterRoot=diameter,
                     dOuterEnd=diameter,
                     dInnerRoot=d_inner,
                     dInnerEnd=d_inner)

        self.m, self.x, self.moi = get_properties([tube,], [Frame(),])



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
                    

#         self.rootFrames.update({'casing':Frame()})
#         self.rootFrames.update({'fore-bulkhead':Frame()})
#         self.rootFrames.update({'aft-bulkhead':Frame(origin=np.array([-geometry['casing-length'] + geometry['casing-thickness'], 0, 0], float))})
#         self.rootFrames.update({'nozzle':Frame(origin=np.array([-geometry['casing-length'], 0, 0], float))})
#         self.rootFrames.update({'grain':Frame(origin=np.array([-geometry['casing-thickness'], 0, 0], float))})

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