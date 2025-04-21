"""Validation tests for primitives are run here"""
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy.linalg import norm

from motion.vectorUtil import ReferenceFrame, drawFrames
from rocket.primitives import *
from ui.textUtil import arrFormat
from rocket.modules import Module
from motion.motionSolvers import linearRK4, angularRK4



def primitiveTest():

    #shape = Conic(length=1, dOuterRoot=1, dOuterEnd=1, dInnerRoot=0, dInnerEnd=0, name="test_conic", material=Aluminium)
    shape = RectangularPrism(x=1, y=1, z=1, material=Aluminium)

    print(f"Name:\t\t{shape.name}")
    print(f"Shape:\t\t{shape.shape}")
    print(f"Material:\t{shape.material.name}")
    print(f"Mass:\t\t{'%.3f' % shape.mass} kg\n")
    
    print(f"Centre of Mass:\n\t\tx: {'%.3f' % shape.com[0]} m\n\t\ty: {'%.3f' % shape.com[1]} m\n\t\tz: {'%.3f' % shape.com[2]} m")
    
    print("\n+===================Inertia Tensors=======================+")
    #use solution found here: https://stackoverflow.com/questions/45478488/python-print-floats-padded-with-spaces-instead-of-zeros

    print(f"Inertia Tensor:\n{arrFormat(shape.moi, sigFigs=3, tabs=2)}")
    
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


def angularTest():

    def freeRotation(t:float, q:np.array, omega:np.array):
        return np.zeros(3, float) # no torque acts on the body in free rotation (in the body frame)

    primitive = Conic(length=10, dOuterRoot=1, dOuterEnd=1, material=Aluminium)
    
    dt = 0.05
    tf = 20
    t = np.arange(0, tf, dt)

    q = np.zeros((t.size, 4), float)
    omega = np.zeros((t.size, 3), float)

    initFrame = ReferenceFrame(axis=np.array([1,0,0], float), ang=0)

    q[0,:] = deepcopy(initFrame.q)
    omega[0,:] = np.array([pi, 0.1, 0])

    for i in range(1, t.size):
        q[i,:], omega[i,:] = angularRK4(q[i-1,:], omega[i-1,:], primitive.moi, t[i], dt, freeRotation)

    # draw the initial cone reference frame:
    objFrame = ReferenceFrame()
    objFrame.q = q[0,:]

    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})

    x = np.array([1,0,0], float)
    y = np.array([0,1,0], float)
    z = np.array([0,0,1], float)

    objX = objFrame.local2parent(x)
    objY = objFrame.local2parent(y)
    objZ = objFrame.local2parent(z)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    linex, = ax.plot([objFrame.translation[0], objX[0]], [objFrame.translation[1], objX[1]], [objFrame.translation[2], objX[2]], '-r')
    liney, = ax.plot([objFrame.translation[0], objY[0]], [objFrame.translation[1], objY[1]], [objFrame.translation[2], objY[2]], '-g')
    linez, = ax.plot([objFrame.translation[0], objZ[0]], [objFrame.translation[0], objZ[1]], [objFrame.translation[2], objZ[2]], '-b')

    ax.set_aspect('equal')

    def update(i):

        objFrame.q = q[i,:]
        
        objX = objFrame.local2parent(x)
        objY = objFrame.local2parent(y)
        objZ = objFrame.local2parent(z)

        linex.set_data([objFrame.translation[0], objX[0]], [objFrame.translation[1], objX[1]])
        liney.set_data([objFrame.translation[0], objY[0]], [objFrame.translation[1], objY[1]])
        linez.set_data([objFrame.translation[0], objZ[0]], [objFrame.translation[0], objZ[1]])

        linex.set_3d_properties([objFrame.translation[2], objX[2]])
        liney.set_3d_properties([objFrame.translation[2], objY[2]])
        linez.set_3d_properties([objFrame.translation[2], objZ[2]])

        # keep the origin of the frame centred in the plot:
        ax.set_xlim([objFrame.translation[0] - 1.2, objFrame.translation[0] + 1.2])
        ax.set_ylim([objFrame.translation[1] - 1.2, objFrame.translation[1] + 1.2])
        ax.set_zlim([objFrame.translation[2] - 1.2, objFrame.translation[2] + 1.2])
        
        ax.set_aspect('equal')

        return linex, liney, linez

    ani = FuncAnimation(fig=fig, func=update, frames=t.size, interval=30)


    fig2, ax2 = plt.subplots()

    # Show how angular momentum decays due to truncation errors:
    # since L = I omega, in the body frame since I is constant, then omega is directly proportional to L.
    magOmega = np.zeros(np.shape(omega)[0], float)
    for i in range(0, np.shape(omega)[0]):
        magOmega[i] = norm(omega[i,:])

    ax2.plot(t, magOmega, '-k')
    ax2.set_xlabel("time, s")
    ax2.set_ylabel("total angular velocity, rad/s")

    
    plt.show()


angularTest()