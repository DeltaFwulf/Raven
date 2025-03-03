"""Validation tests for primitives are run here"""
from math import floor, log10
from copy import deepcopy

from motion.vectorUtil import Transform, drawFrames
from rocket.primitives import *
from ui.textUtil import arrFormat
from rocket.modules import Module



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

    print(f"Mass:\n{arrFormat(shape.moi_com, sigFigs=3, tabs=2)}")
    print(f"Root:\n{arrFormat(shape.moi_root, sigFigs=3, tabs=2)}")
    print(f"Reference:\n{arrFormat(shape.moi_root, sigFigs=3, tabs=2)}")

    # TODO: draw the shape in the module frame


def frameTest():
    """Uses the Transform class in different ways and outputs the result in a 3D plot to validate transform behaviour
    
    Passed
    """

    baseFrame = Transform() # this is just a trivial transform (no change from "true" origin)

    move1 = deepcopy(baseFrame)
    move1.move(axis=np.array([0,1,0]), ang=pi/6, translation=np.array([1,0,0]))

    move2 = deepcopy(move1)
    move2.move(axis=np.array([0,0,1]), ang=pi, translation=np.array([0,0,1]))

    move3 = deepcopy(move1)
    move3.move(axis=np.array([0,0,1]), ang=pi, translation = np.array([3,0,0]), reference='parent')

    drawFrames([baseFrame, move1, move2, move3])



def chainTest():
    """Chains multiple transformations together to make sure they behave as expected
    
    Passed
    """

    baseFrame = Transform()

    move1 = Transform(axisInit=np.array([1,0,0]), angInit=pi/6, transInit=np.array([5,0,0]))
    move2 = Transform(transInit=np.array([0,1,0]))
    
    chain = deepcopy(move1)
    chain.chain(move2)

    drawFrames([baseFrame, move1, chain])



def moduleTest():
    """
    A module is built and its properties output, to be checked against the same geometry in CAD.
    
    Findings:
    - we need a good way to access the primitive and their transform by a name, or to associate an index with a useful name so that we can assign behaviour to certain primitives

    Passed
    """

    # Where within the Module is the primitive?
    cylinder = Conic(length=1, dOuterRoot=1, dOuterEnd=1, dInnerRoot=0, dInnerEnd=0, name='cylinder', material=Aluminium)
    cone = Conic(length=0.5, dOuterRoot=1, dOuterEnd=0, dInnerRoot=0, dInnerEnd=0, name='cone', material=Aluminium)

    cylinderTransform = Transform(axisInit=np.array([0,1,0], float), angInit=pi/4)
    coneTransform = deepcopy(cylinderTransform)
    coneTransform.move(translation=np.array([1,0,0], float), reference='local')
    #coneTransform = Transform(transInit=np.array([1.5,0,0]), axisInit=np.array([0,1,0]), angInit=pi)

    primitives = [cylinder, cone]
    rootTransforms = [cylinderTransform, coneTransform]

    for i in range(0, len(primitives)):
        print("=========================================")
        print(f"Name: {primitives[i].name}")
        print(f"Mass:\n\t\t{'%.3f' % primitives[i].mass} kg\n")
        print(f"Centre of Mass:\n{arrFormat(primitives[i].com, sigFigs=3, tabs=2)}\n")
        print(f"Moment of Inertia Tensor:\n{arrFormat(primitives[i].moi, sigFigs=3, tabs=2)}\n")

    module = Module(primitives=primitives, rootTransforms=rootTransforms)

    # get the module's mass, CoM, MoI tensor, compare against result in CAD and hand calc:
    print(f"Module mass:\n\t\t{'%.3f' % module.mass} kg")
    print(f"Module CoM:\n{arrFormat(module.com, sigFigs=3, tabs=2)}\n")
    print(f"Module MoI:\n{arrFormat(module.moi, sigFigs=3, tabs=2)}\n")

#shapeTester()
#frameTest()
#chainTest()
moduleTest()