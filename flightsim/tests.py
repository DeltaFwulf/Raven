"""Validation tests for primitives are run here"""
from math import floor, log10
from motion.transforms import Transform, drawFrames
from rocket.primitives import *
from ui.textutil import formatArray


def shapeTester():

    transform = Transform(transInit=np.array([0,0,0], float), angInit=pi/4, axisInit=np.array([1,0,0], float))
    shape = Conic(length=1, moduleTransform=transform, dOuterRoot=1, dOuterEnd=0, dInnerRoot=0.5, dInnerEnd=0, name="test_conic", material=Aluminium)

    print(f"Name:\t\t{shape.name}")
    print(f"Shape:\t\t{shape.shape}")
    print(f"Material:\t{shape.material.name}")
    print(f"Mass:\t\t{'%.3f' % shape.mass} kg\n")
    
    print(f"Centre of Mass:\n\t\tx: {'%.3f' % shape.com[0]} m\n\t\ty: {'%.3f' % shape.com[1]} m\n\t\tz: {'%.3f' % shape.com[2]} m")
    
    print("\n+===================Inertia Tensors=======================+")
    # TODO: identify the number of spaces required by the largest number's order of magnitude, use solution found here: https://stackoverflow.com/questions/45478488/python-print-floats-padded-with-spaces-instead-of-zeros

    sigFigs = 3
    sf = "%." + str(sigFigs) + "f"

    def getMaxOrder(array:np.array) -> int:

        maxMag = 0
        nonZeros = array[array != 0]
        
        for i in range(0, nonZeros.size):
            thisMag = floor(log10(nonZeros[i]))
            if thisMag > maxMag:
                maxMag = thisMag

        return maxMag

    # find order of magnitude of largest non-zero element of inertia tensors:
    maxMagCom = getMaxOrder(shape.moi_com)
    maxMagRoot = getMaxOrder(shape.moi_root)
    maxMagRef = getMaxOrder(shape.moi_ref)

    maxMag = max(maxMagCom, maxMagRoot, maxMagRef)
    pad = "{0: >" + str(maxMag + 2 + sigFigs) + "}"
    
    print(f"Mass:\n{formatArray(shape.moi_com, sigFigs=3, tabs=2)}")
    print(f"Root:\n{formatArray(shape.moi_root, sigFigs=3, tabs=2)}")
    print(f"Reference:\n{formatArray(shape.moi_root, sigFigs=3, tabs=2)}")

    # TODO: draw the shape in the module frame

def frameTest():
    """Uses the Transform class in different ways and outputs the result in a 3D plot to validate transform behaviour"""

    baseFrame = Transform() # this is just a trivial transform (no change from "true" origin)

    # let's set the first transformation to be 45 degrees about the x axis:
    transform1 = Transform(transInit=np.array([1,1,1], float), angInit=pi/4, axisInit=np.array([1,0,0], float))

    # then, we'll transform this by moving in the new x axis by 5 and rotating 180 degrees about the y axis:
    transform2 = Transform(transInit=np.array([0,0,0], float), angInit=pi, axisInit=np.array([0,1,0], float))
    transform2.chain(transform1)

    # can we then translate frame 2 by 2 in its local z axis?
    transform2.transformLocal(np.array([0,0,2], float))

    frames = [baseFrame, transform1, transform2]
    drawFrames(frames)



shapeTester()
#frameTest()