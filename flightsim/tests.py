"""A script containing several testing functions"""

from motion.frame import Frame
from rocket.primitives import *



def shapeTester():

    transform = Frame(transInit=np.array([0,0,0], float), angInit=0, axisInit=np.array([0,1,0], float))
    testPrimitive = Conic(length=1, transform=transform, dOuterRoot=1, dOuterEnd=0, dInnerRoot=0, dInnerEnd=0, name="conic", material="generic")

    # Primitive properties:
    ###########################################################################################################################################################
    print(f"the mass of {testPrimitive.name} is {testPrimitive.mass} kg")
    print(f"the centre of mass of {testPrimitive.name} is\n{testPrimitive.com}")
    print(f"mass inertia tensor:\n\n{testPrimitive.moi_com}\n")
    print(f"root inertia tensor:\n\n{testPrimitive.moi_root}\n")
    print(f"ref inertia tensor:\n\n{testPrimitive.moi_ref}\n")



shapeTester()