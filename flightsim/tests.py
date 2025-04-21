"""Validation tests for primitives are run here"""
from copy import deepcopy
import matplotlib.pyplot as plt

from motion.vectorUtil import ReferenceFrame, drawFrames
from rocket.primitives import *
from ui.textUtil import arrFormat
from rocket.modules import Module
from motion.motionSolvers import linearRK4



def shapeTester():

    transform = ReferenceFrame(transInit=np.array([0,0,0], float), angInit=pi/4, axisInit=np.array([1,0,0], float))
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

    worldFrame = ReferenceFrame() # this is just a trivial transform (no change from "true" origin)

    frame1 = deepcopy(worldFrame)
    frame1.move(axis=np.array([0,1,0]), ang=pi/6, translation=np.array([1,0,0]))

    frame2 = deepcopy(frame1)
    frame2.move(axis=np.array([0,0,1]), ang=pi, translation=np.array([0,0,1]))

    frame3 = deepcopy(frame1)
    frame3.move(axis=np.array([0,0,1]), ang=pi, translation = np.array([3,0,0]), reference='parent')

    drawFrames([worldFrame, frame1, frame2, frame3])



def chainTest():
    """Chains multiple transformations together to make sure they behave as expected
    
    Passed
    """

    worldFrame = ReferenceFrame()

    frame1 = ReferenceFrame(axisInit=np.array([1,0,0]), angInit=pi/6, transInit=np.array([5,0,0]))
    frame2 = ReferenceFrame(transInit=np.array([0,1,0]))
    
    chain = deepcopy(frame1)
    chain.chain(frame2)

    drawFrames([worldFrame, frame1, chain])



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

    cylinderTransform = ReferenceFrame(axis=np.array([0,1,0], float), ang=0)
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



def linearTest():
    """Tests the linear RK4 motion function by simulating the trajectory of a primitive thrown upwards on the Earth."""

    def gravity(X:np.array, dt:float, params:dict) -> np.array:
        return -params['mu'] * X[0,:] / np.linalg.norm(X[0,:])**3
    
    primitive = Conic(length=1.0, dOuterRoot=1, dOuterEnd=1, material=Aluminium)
    params = {'mu':3.986e14, 'object':primitive}

    dt = 0.1
    tf = 20
    t = np.arange(0, tf, dt)
    
    x = np.zeros((t.size, 3), float)
    v = np.zeros((t.size, 3), float)

    x[0,:] = [6378e3, 0, 0]
    v[0,:] = [100, 0, 0]

    for i in range(1, t.size):
        x[i,:], v[i,:] = linearRK4(x[i-1], v[i-1], dt, gravity, params)

    fig, ax = plt.subplots()

    ax.plot(t, x[:,0], '-')
    ax.set_xlabel('time, s')
    ax.set_label('x, m')
    plt.show()



moduleTest()