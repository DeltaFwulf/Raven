import numpy as np
from numpy.linalg import norm
from math import pi, cos, sin, atan2, asin, isnan
from copy import deepcopy



def cartesian2spherical(vector:np.array) -> tuple[float, float, float]:
    """Converts a cartesian vector to spherical coordinates (r, inc, az) following ISO convention"""
    r = norm(vector)

    inc = atan2(norm(vector[0:-1]), vector[2])
    if isnan(inc):
        inc = 0

    az = atan2(vector[1], vector[0])
    if isnan(az):
        az = 0

    return r, inc, az


def cartesian2coords(vector:np.array) -> tuple['float', 'float']:
    """ Calculates the latitude and longitude on a spherical body given a cartesian vector"""
    lat = pi / 2 - atan2(norm(vector[0:-1]), vector[2])
    if isnan(lat):
        lat = pi / 2

    long = atan2(vector[1], vector[0])
    if isnan(long):
        long = 0

    return lat, long


def coords2cartesian(lat:float, long:float, r:float) -> np.array:
    """Returns a cartesian vector given latitude, longitude, and radius"""
    x = r*cos(lat)*cos(long)
    y = r*cos(lat)*sin(long)
    z = r*sin(lat)

    return np.array([x, y, z], float)


def sphereAngs2cartesian(inc:float, az:float, r:float) -> np.array:
    """Returns the cartesian vector given spherical coordinates"""
    x = r*sin(inc)*cos(az)
    y = r*sin(inc)*sin(az)
    z = r*cos(inc)

    return np.array([x, y, z], float)


def coords2sphereAngs(latitude, longitude) -> tuple[float, float]:
    """Converts latitude and longitude into spherical inclination and azimuth"""
    return (pi/2) - latitude, longitude


def sphereAngs2coords(inclination:float, azimuth:float) -> tuple[float, float]:
    latitude = pi/2 - inclination
    return latitude, azimuth


def getAngleSigned(vecA:np.array, vecB:np.array, normal:np.array) -> float:
    """Gets the angle from vecA to vecB in the correct direction, given that they both lie on a known plane."""
    return atan2(np.dot(unit(normal), np.cross(vecA, vecB)), np.dot(vecA, vecB))


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


def grassmann(qA, qB):
    """Returns the inner product of two quaternions (similar to dot-product but with orthogonal vectors)"""

    qOut = np.zeros(4, float)

    qOut[0] = qA[0] * qB[0] - np.dot(qA[1:],qB[1:])
    qOut[1:] = qA[0]*qB[1:] + qA[1:]*qB[0] + np.cross(qA[1:], qB[1:])

    return qOut


def rotateQuaternion(vecIn, q):
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    qConv = deepcopy(q)
    qConv[1:] *= -1.0

    t = np.cross(2*q[1:], vecIn)
    return(vecIn + q[0]*t + np.cross(q[1:], t))     


def unit(vec:np.array) -> np.array:
    """Returns a unit vector pointed in the same direction as the original vector""" 
    return vec / norm(vec)



def rotateAxisAngle(vecIn:np.array, axis:np.array, ang:float) -> np.array:

    axis = unit(axis)
    q = np.zeros(4, float)
    q[0] = cos(ang / 2)
    q[1:] = sin(ang / 2)*axis

    return rotateQuaternion(vecIn, q)


def axisAngle2Quaternion(axis:np.array, angle:float) -> np.array:
        q = np.zeros(4, float)
        q[0] = cos(angle / 2)
        q[1:] = sin(angle/ 2)*axis

        return q / norm(q)


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
    
    return axisAngle2Quaternion(rotAxis, getAngleUnsigned(referenceAxis, newAxis))