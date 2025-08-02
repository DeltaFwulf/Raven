import numpy as np
from numpy.linalg import norm
from math import pi, cos, sin, atan2, asin, sqrt, isnan
import matplotlib.pyplot as plt
from copy import deepcopy



class ReferenceFrame():
    """
    This class represents the transformation between two reference frames. Use this class to map vectors or points between different axis systems.
    Also contains functions that can be used to map inertia tensors between axis systems according to the frame transform.

    Use this class to create reference frames, move reference frames, or map vectors into different frames of reference.
    """

    def __init__(self, axis:np.array=None, ang:float=None, translation:np.array=np.zeros(3,float), sphereAngs:np.array=None, roll:float=None, axisName:str=None):

        """
        Passing Spherical Coordinates with Roll:
            You can pass an array representing inclination and azimuth on the unit sphere (aligned to parent frame), with an explicit roll angle.
            In this case, the unit vector from the sphere origin and the location on the sphere is the new axis (named either 'x', 'y', or 'z').
            The transformation between the parent 'x', 'y', or 'z' is determined and sets the initial rotation matrix. The roll is then applied in the new frame's local system to get the final rotation matrix.
        """

        self.q = np.array([1,0,0,0], float)
        self.translation = translation

        if axis is not None:
            self.q = ReferenceFrame.axisAngle2Quaternion(axis, ang)
        
        elif sphereAngs is not None:
            
            referenceAxis = np.zeros(3, float)

            match axisName:
                case 'x':
                    n = 0
                case 'y':
                    n = 1
                case 'z':
                    n = 2

            referenceAxis[n] = 1

            # what is the new axis?
            newAxis = np.zeros(3, float)
            newAxis[0] = sin(sphereAngs[0]) * cos(sphereAngs[1])
            newAxis[1] = sin(sphereAngs[0]) * sin(sphereAngs[1])
            newAxis[2] = cos(sphereAngs[0])

            # get the axis and angle of rotation
            if (newAxis == referenceAxis).all():
                rotAxis = np.array([1,0,0], float)
            else:
                rotAxis = np.cross(referenceAxis, newAxis)
                rotAxis /= norm(rotAxis) # normalise the vector
            
            angle = getAngleUnsigned(referenceAxis, newAxis)

            self.q = ReferenceFrame.axisAngle2Quaternion(rotAxis, angle)
            self.move(referenceAxis, roll) # apply roll


    def axisAngle2Quaternion(axis:np.array, angle:float):

        q = np.zeros(4, float)
        q[0] = cos(angle / 2)
        q[1:] = sin(angle/ 2) * axis

        return q / norm(q)


    def quat2RotationMatrix(q:np.array) -> np.array:
        """Given a unit quaternion, outputs a 3x3 rotation matrix"""

        rotMatrix = np.zeros((3,3), float)

        rotMatrix[0,0] = 2 * (q[0]**2 + q[1]**2) - 1
        rotMatrix[0,1] = 2 * (q[1]*q[2] - q[0]*q[3])
        rotMatrix[0,2] = 2 * (q[1]*q[3] + q[0]*q[2])
        
        rotMatrix[1,0] = 2 * (q[1]*q[2] + q[0]*q[3])
        rotMatrix[1,1] = 2 * (q[0]**2 + q[2]**2) - 1
        rotMatrix[1,2] = 2 * (q[2]*q[3] - q[0]*q[1])

        rotMatrix[2,0] = 2 * (q[1]*q[3] - q[0]*q[2])
        rotMatrix[2,1] = 2 * (q[2]*q[3] + q[0]*q[1])
        rotMatrix[2,2] = 2 * (q[0]**2 + q[3]**2) - 1

        return rotMatrix
    

    def sphereAngs2Quaternion(sphereAngs:list[float], axis:str='x') -> np.array:

        referenceAxis = np.zeros(3, float)

        match axis:
            case 'x':
                n = 0
            case 'y':
                n = 1
            case 'z':
                n = 2

        referenceAxis[n] = 1

        # what is the new axis?
        newAxis = np.zeros(3, float)
        newAxis[0] = sin(sphereAngs[0]) * cos(sphereAngs[1])
        newAxis[1] = sin(sphereAngs[0]) * sin(sphereAngs[1])
        newAxis[2] = cos(sphereAngs[0])

        # get the axis and angle of rotation
        if (newAxis == referenceAxis).all():
            rotAxis = np.array([1,0,0], float)
        else:
            rotAxis = np.cross(referenceAxis, newAxis)
            rotAxis /= norm(rotAxis) # normalise the vector
        
        #angle = acos(np.dot(referenceAxis, newAxis) / (norm(referenceAxis) * norm(newAxis)))
        return ReferenceFrame.axisAngle2Quaternion(rotAxis, getAngleUnsigned(referenceAxis, newAxis))

    

    def move(self, axis:np.array=np.array([1,0,0], float), ang:float=0, translation:np.array=np.array([0,0,0], float), reference:str='local') -> None:
        """Moves the reference frame according to a rotation and translation, in either local or parent frame's reference."""

        axis /= norm(axis) # normalise the axis (so that we can pass any axis of rotation into the function)

        if reference == 'parent':

            qConv = deepcopy(self.q)
            qConv[1:] *= -1.0
            
            axis = rotateQuaternion(axis, qConv)

            q = np.array([cos(ang/2), axis[0]*sin(ang/2), axis[1]*sin(ang/2), axis[2]*sin(ang/2)], float)
            q /= norm(q) # renormalise q here to prevent drift

            self.q = grassmann(self.q, q)
            self.translation += translation

        elif reference == 'local':
            transform = np.identity((4))

            q = np.array([cos(ang/2), axis[0]*sin(ang/2), axis[1]*sin(ang/2), axis[2]*sin(ang/2)], float)
            q /= norm(q) # prevent drift

            # chain the rotations, but apply the translation within the original frame
            translation = self.local2parent(translation, incTranslation=False)
            self.q = grassmann(self.q, q)
            self.translation += translation

        else:
            print("reference keyword invalid: please use either local or parent")


    def invert(self) -> None:
        """Inverts the transformation"""
        self.q[1:] *= -1.0
        self.translation = -self.translation


    def local2parent(self, vecIn:np.array, incTranslation:bool=True) -> np.array:
        """If the vector is described in the local coordinate system, this returns the same vector as expressed in the world coordinate system
        
        if incTranslation == False, this purely rotates the vector (assumes the vector passed in was from the parent frame origin)
        """
        
        vecOut = rotateQuaternion(vecIn, self.q)
        
        if incTranslation:
            vecOut += self.translation

        return vecOut

       
    def parent2local(self, vecIn:np.array, incTranslation:bool=True) -> np.array:
        """If the vector is specified in world coordinates, this is what the vector would be expressed as in local coordinates
        
        if incTranslation == False, this purely rotates the vector (assumes the localVector shared an origin with the local frame origin)
        """

        if incTranslation:
            vecIn -= self.translation

        qConv = deepcopy(self.q)
        qConv[1:] *= -1
        return rotateQuaternion(vecIn, qConv)
        # return np.matmul(self.rotationMatrix.transpose(), vecIn)
    

    # TODO: deprecate this, it's covered by local2parent
    def align(self, vecIn:np.array) -> np.array:
        """Maps a vector defined in this reference frame into the world frame BUT assumes that this reference frame shares an origin with the parent - pure rotation without translation"""
        return rotateQuaternion(vecIn, self.q)
        #return np.matmul(self.rotationMatrix(), vecIn)
    
    
    def chain(self, nextFrame:'ReferenceFrame'):
            """This combines two subsequent transformations together, in the order of self.transform, newTransform"""

            chainedRotation = grassmann(self.q, nextFrame.q)

            self.q = chainedRotation
            self.translation += nextFrame.translation


    def transformInertiaTensor(self, tensorIn, mass, com2ref:np.array=np.zeros(3)) -> np.array:
        """Changes the reference frame of the mass moment of inertia tensor to that about a reference frame with this transform from the body aligned, CoM centered reference frame.
        
        For generalised parallel axis theorem, you can also specify translation (in parent coordinates) of the reference location from the object's centre of mass. If the tensor input is about the object's centre of mass,
        do not put anything for initialTranslation. If the initial translation is non-zero, please put the translation in; the generalised parallel axis theorem method can take this into account.
        """
        
        rotating = self.q[0] != 1
        translating = (self.translation != np.zeros((3), float)).any()

        tensor = tensorIn

        if rotating:
            
            i = np.array([1, 0, 0])
            j = np.array([0, 1, 0])
            k = np.array([0, 0, 1])

            iNew = self.align(i)
            jNew = self.align(j)
            kNew = self.align(k)

            def cosAng(vec1, vec2):
                return np.dot(vec1, vec2) / sqrt(np.linalg.norm(vec1) * np.linalg.norm(vec2))
            T = np.empty((3,3), float)
        
            T[0,0] = cosAng(iNew, i)
            T[0,1] = cosAng(iNew, j)
            T[0,2] = cosAng(iNew, k)

            T[1,0] = cosAng(jNew, i)
            T[1,1] = cosAng(jNew, j)
            T[1,2] = cosAng(jNew, k)

            T[2,0] = cosAng(kNew, i)
            T[2,1] = cosAng(kNew, j)
            T[2,2] = cosAng(kNew, k)

            tensor = np.matmul(T, np.matmul(tensor, np.transpose(T)))

        if translating:
            """
            This function uses a generalised form of the parallel axis theorem found here: https://doi.org/10.1119/1.4994835

            I' = Iref + M[(R2,R2)] - 2M[(R2,C)]
            """

            def getSymmetricMatrix(a, b):

                c = np.empty((3,3), float)

                c[0,0] = a[1]*b[1] + a[2]*b[2]
                c[0,1] = -0.5 * (a[0]*b[1] + a[1]*b[0])
                c[0,2] = -0.5 * (a[0]*b[2] + a[2]*b[0])

                c[1,0] = c[0,1]
                c[1,1] = a[0]*b[0] + a[2]*b[2]
                c[1,2] = -0.5 * (a[1]*b[2] + a[2]*b[1])

                c[2,0] = c[0,2]
                c[2,1] = c[1,2]
                c[2,2] = a[0]*b[0] + a[1]*b[1]

                return c

            translation = self.translation

            tensor += (mass * getSymmetricMatrix(translation, translation)) - (2 * mass * getSymmetricMatrix(translation, com2ref))
                                                
            return tensor
        


def cartesian2spherical(vector:np.array) -> tuple[float, float, float]:
    """
    Converts a cartesian vector to spherical coordinates (r, inc, az)
    here we are following the ISO (physics convention)
    """
    
    r = norm(vector)

    inc = atan2(norm(vector[0:-1]), vector[2])
    if isnan(inc):
        inc = 0

    az = atan2(vector[1], vector[0])
    if isnan(az):
        az = 0

    return r, inc, az



def coords2sphereAngs(latitude, longitude) -> tuple[float, float]:
    """Converts latitude and longitude into spherical inclination and azimuth
    
    we are following the ISO (physics angle convention)
    """
    
    inclination = (pi/2) - latitude
    return inclination, longitude



def sphereAngs2coords(inclination:float, azimuth:float) -> tuple[float, float]:

    latitude = pi/2 - inclination
    return latitude, azimuth



def getAngleSigned(vecA:np.array, vecB:np.array, planeNormal:np.array) -> float:

    """Gets the angle from vecA to vecB in the correct direction, given that they both lie on a known plane."""

    n = planeNormal / norm(planeNormal)
    return atan2(np.dot(n, np.cross(vecA, vecB)), np.dot(vecA, vecB))



def getAngleUnsigned(vecA:np.array, vecB:np.array) -> float:
    """Returns the magnitude of the angle between two vectors, but cannot give the sign (direction) of the angle."""
    return atan2(norm(np.cross(vecA, vecB)), np.dot(vecA, vecB))



def projectVector(vecA:np.array, vecB:np.array, comp:str) -> np.array:
    """Projects vector A along vector B, returns either 'parallel' or 'normal' component of vector A to vector B."""
    bUnit = vecB / norm(vecB)
    parallel = np.dot(vecA, bUnit) * bUnit

    if comp == 'parallel':
        return parallel
    
    else:
        return vecA - parallel
    

def quaternion2euler(q:float, order:str='tait-bryan'):
    
    if order=='tait-bryan':
        # flight dynamics convention (heading, pitch, bank)
        roll = atan2(2*(q[0]*q[1] + q[2]*q[3]), 1- 2*(q[1]**2 + q[2]**2))
        pitch = asin(2*(q[0]*q[2] - q[1]*q[3]))
        yaw = atan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))

    return roll, pitch, yaw


def grassmann(a, b):
    """direct quaternion multiplication, symmetric product"""

    qOut = np.zeros(4, float)

    qOut[0] = a[0] * b[0] - np.dot(a[1:],b[1:])
    qOut[1:] = a[0]*b[1:] + a[1:]*b[0] + np.cross(a[1:], b[1:])

    return qOut


def rotateQuaternion(vecIn, q):

    # TODO: use this faster algorithm for quaternion rotation:
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

        qConv = deepcopy(q)
        qConv[1:] *= -1.0

        t = np.cross(2*q[1:], vecIn)
        return(vecIn + q[0]*t + np.cross(q[1:], t))

        #return grassmann(grassmann(q, np.hstack((0, vecIn))), qConv)[1:]       
        


def drawFrames(frames:list[ReferenceFrame]) -> None:

    """This function takes in a list of frames and for each one plots a set of orthogonal axes according to their respective transforms."""

    ax = plt.figure().add_subplot(projection='3d')

    xMin = frames[0].translation[0]
    xMax = frames[-1].translation[0]
    yMin = frames[0].translation[1]
    yMax = frames[-1].translation[1]
    zMin = frames[0].translation[2]
    zMax = frames[-1].translation[2]

    for frame in frames:

        x = frame.local2parent(np.array([1,0,0]))
        y = frame.local2parent(np.array([0,1,0]))
        z = frame.local2parent(np.array([0,0,1]))
        o = frame.translation

        # update plot limits
        xMin = np.min(np.array([xMin, x[0], y[0], z[0]]))
        xMax = np.max(np.array([xMax, x[0], y[0], z[0]]))

        yMin = np.min(np.array([yMin, x[1], y[1], z[1]]))
        yMax = np.max(np.array([yMax, x[1], y[1], z[1]]))

        zMin = np.min(np.array([zMin, x[2], y[2], z[2]]))
        zMax = np.max(np.array([zMax, x[2], y[2], z[2]]))
        
        ax.plot([o[0], x[0]], [o[1], x[1]], [o[2], x[2]], '-r')
        ax.plot([o[0], y[0]], [o[1], y[1]], [o[2], y[2]], '-g')
        ax.plot([o[0], z[0]], [o[1], z[1]], [o[2], z[2]], '-b')

    # dynamically bound the plot based on the largest values of any terms in x, y, z
    ax.set_xlim([xMin, xMax])
    ax.set_ylim([yMin, yMax])
    ax.set_zlim([zMin, zMax])

    ax.set_box_aspect([xMax - xMin, yMax - yMin, zMax - zMin])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.legend(['x', 'y', 'z'])

    plt.show()



def frameTest():
    """Move a frame in different ways and plot to visually verify the results"""

    rootFrame = ReferenceFrame(axis=np.array([1,0,0], float), ang=0, translation=np.array([0,0,0], float))

    testFrame = ReferenceFrame(axis=np.array([1,1,0], float), ang=10*pi/180, translation=np.array([1,0,0], float))
    globalMoved = deepcopy(testFrame)
    localMoved = deepcopy(testFrame)
    inverted = deepcopy(testFrame)
    chained = deepcopy(testFrame)
    globalMoved.move(axis=np.array([0,0,1], float), ang=pi, translation=np.array([1,0,0], float), reference='parent')
    localMoved.move(axis=np.array([0,0,1], float), ang=pi, translation=np.array([1,0,0], float), reference='local')
    inverted.invert()
    chained.chain(testFrame)

    toAlign = np.array([1,0,0], float)

    aligned = chained.align(toAlign)
    print(f"local2parent: {chained.local2parent(toAlign)}, aligned: {aligned}")

    drawFrames([rootFrame, testFrame, globalMoved, localMoved])



def conversionTest():

    cart = np.array([-1, -0.1, -0.5])
    print(f"x:{cart[0]}, y:{cart[1]}, z:{cart[2]}")

    radius, inclination, azimuth = cartesian2spherical(cart)
    print(f"radius: {radius}, inclination: {inclination * 180 / pi}, azimuth: {azimuth * 180 / pi}")

    latitude, longitude = sphereAngs2coords(inclination, azimuth)
    print(f"latitude: {latitude * 180 / pi}, longitude {longitude * 180 / pi}")

    inc2, az2 = coords2sphereAngs(latitude, longitude)
    print(f"returned inclination: {inc2 * 180 / pi}, returned azimuth: {az2 * 180 / pi}")



def rotationTests():

    axis = np.array([1,0,0], float)
    ang = pi/2

    q1 = np.zeros(4, float)
    q1[0] = cos(ang / 2)
    q1[1:] = sin(ang / 2) * axis

    v = np.array([0,1,0], float)

    frame = ReferenceFrame(axis=axis, ang=ang)

    vRotatedFrame = frame.local2parent(v)
    vSingle = rotateQuaternion(v, q1)

    # Perform a 90 degree rotation by chaining two successive 45 degree rotations:
    q2 = grassmann(q1, q1)
    vDouble = rotateQuaternion(v, q2)

    q3 = deepcopy(q1)
    q3[1:] *= -1

    vOppo = rotateQuaternion(v, q3)

    print(f"\n Rotation Matrix: {vRotatedFrame}")
    print(f"\nSingle: {vSingle}")
    print(f"\nDouble: {vDouble}")
    print(f"\nOpposite: {vOppo}")