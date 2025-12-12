import numpy as np
from numpy.linalg import norm
from math import pi, cos, sin, atan2, asin, isnan
from copy import deepcopy



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


def cartesian2coords(vector:np.array) -> tuple['float', 'float']:

    inc = atan2(norm(vector[0:-1]), vector[2])
    if isnan(inc):
        inc = 0

    az = atan2(vector[1], vector[0])
    if isnan(az):
        az = 0

    return (pi / 2) - inc, az


def coords2cartesian(lat:float, long:float, r:float) -> np.array:
    """Returns the position vector given latitude, longitude, and radius"""
    x = r*cos(lat)*cos(long)
    y = r*cos(lat)*sin(long)
    z = r*sin(lat)

    return np.array([x, y, z], float)


def sphereAngs2cartesian(inc:float, az:float, r:float) -> np.array:
    """Returns the position vector given spherical coordinates"""
    x = r*sin(inc)*cos(az)
    y = r*sin(inc)*sin(az)
    z = r*cos(inc)

    return np.array([x, y, z], float)


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
    return atan2(np.dot(unit(planeNormal), np.cross(vecA, vecB)), np.dot(vecA, vecB))


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