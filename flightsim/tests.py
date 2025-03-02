"""Validation tests for primitives are run here"""
from math import floor, log10
from motion.vectorUtil import Transform, drawFrames
from rocket.primitives import *
from ui.textUtil import formatArray
from copy import deepcopy


def shapeTester():

    transform = Transform(transInit=np.array([0,0,0], float), angInit=pi/4, axisInit=np.array([1,0,0], float))
    shape = Conic(length=1, moduleTransform=transform, dOuterRoot=1, dOuterEnd=0, dInnerRoot=0.5, dInnerEnd=0, name="test_conic", material=Aluminium)

    print(f"Name:\t\t{shape.name}")
    print(f"Shape:\t\t{shape.shape}")
    print(f"Material:\t{shape.material.name}")
    print(f"Mass:\t\t{'%.3f' % shape.mass} kg\n")
    
    print(f"Centre of Mass:\n\t\tx: {'%.3f' % shape.com[0]} m\n\t\ty: {'%.3f' % shape.com[1]} m\n\t\tz: {'%.3f' % shape.com[2]} m")
    
    print("\n+===================Inertia Tensors=======================+")
    #use solution found here: https://stackoverflow.com/questions/45478488/python-print-floats-padded-with-spaces-instead-of-zeros

    print(f"Mass:\n{formatArray(shape.moi_com, sigFigs=3, tabs=2)}")
    print(f"Root:\n{formatArray(shape.moi_root, sigFigs=3, tabs=2)}")
    print(f"Reference:\n{formatArray(shape.moi_root, sigFigs=3, tabs=2)}")

    # TODO: draw the shape in the module frame


def frameTest():
    """Uses the Transform class in different ways and outputs the result in a 3D plot to validate transform behaviour"""

    baseFrame = Transform() # this is just a trivial transform (no change from "true" origin)

    move1 = deepcopy(baseFrame)
    move1.move(axis=np.array([0,1,0]), ang=pi/6, translation=np.array([1,0,0]))

    move2 = deepcopy(move1)
    move2.move(axis=np.array([0,0,1]), ang=pi, translation=np.array([0,0,1]))

    move3 = deepcopy(move1)
    move3.move(axis=np.array([0,0,1]), ang=pi, translation = np.array([3,0,0]), reference='parent')

    drawFrames([baseFrame, move1, move2, move3])



def chainTest():
    """Chains multiple transformations together to make sure they behave as expected"""

    baseFrame = Transform()

    move1 = Transform(axisInit=np.array([1,0,0]), angInit=pi/6, transInit=np.array([5,0,0]))
    move2 = Transform(transInit=np.array([0,1,0]))
    
    chain = deepcopy(move1)
    chain.chain(move2)

    drawFrames([baseFrame, move1, chain])



def tensorTests():
    """Moves around primitives, then forms a compound object from primitives, returns the primitives and compound tensors"""

    # Where within the Module is the primitive?
    primitive0 = Conic(moduleTransform=Transform(), length=1, dOuterRoot=1, dOuterEnd=1, dInnerRoot=0, dInnerEnd=0, name='conic0', material=Aluminium)
    



#shapeTester()
#frameTest()
chainTest()