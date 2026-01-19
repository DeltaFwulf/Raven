import numpy as np
from numpy.linalg import norm
from math import pi, cos, sin, atan2, isnan



def cartesian2spherical(vector:np.ndarray) -> tuple[float, float, float]:
    """Converts a cartesian vector to spherical coordinates (r, inc, az) following ISO convention"""
    r = norm(vector)

    inc = atan2(norm(vector[0:-1]), vector[2])
    if isnan(inc):
        inc = 0

    az = atan2(vector[1], vector[0])
    if isnan(az):
        az = 0

    return r, inc, az


def spherical2cartesian(inc:float, az:float, r:float) -> np.ndarray:
    """Returns the cartesian vector given spherical coordinates"""

    if inc%(2*pi) > pi:
        inc = 2*pi - inc%(2*pi)
        az += pi

    return r*np.r_[sin(inc)*cos(az), sin(inc)*sin(az), cos(inc)]

   
def cartesian2coords(vector:np.ndarray) -> tuple['float', 'float']:
    """ Calculates the latitude and longitude on a spherical body given a cartesian vector"""
    lat = pi / 2 - atan2(norm(vector[0:-1]), vector[2])
    if isnan(lat):
        lat = pi / 2

    long = atan2(vector[1], vector[0])
    if isnan(long):
        long = 0

    return lat, long


def coords2cartesian(lat:float, long:float, r:float) -> np.ndarray:
    """Returns a cartesian vector given latitude, longitude, and radius"""
    x = r*cos(lat)*cos(long)
    y = r*cos(lat)*sin(long)
    z = r*sin(lat)

    return np.array([x, y, z], float)


def coords2spherical(lat, long) -> tuple[float, float]:
    """Converts latitude and longitude into spherical inclination and azimuth"""

    inc = pi / 2 - lat
    az = long

    if inc % (2*pi) > pi:
        inc = 2*pi - inc % (2*pi)
        az += pi

    return inc, az % (2*pi)


def spherical2coords(inc:float, az:float) -> tuple[float, float]:
    """Returns latitude and longitude (East) given inclination and azimuth"""

    if inc % (2*pi) > pi:
        inc = 2*pi - inc % (2*pi)
        az += pi

    if az % (2*pi) > pi:
        az -= 2*pi

    return pi / 2 - inc, az


def getAngleSigned(vecA:np.ndarray, vecB:np.ndarray, normal:np.ndarray) -> float:
    """Gets the angle from vecA to vecB in the correct direction, given that they both lie on a known plane."""

    if norm(vecA) == 0 or norm(vecB) == 0 or norm(normal) == 0:
        raise ValueError
    
    vecA /= norm(vecA)
    vecB /= norm(vecB)
    normal /= norm(normal)
    
    if norm((vecA + vecB)*normal) > 1e-12: # vectors must be coplanar and no vector may have zero-length
        raise ValueError

    return atan2(np.dot(np.cross(vecA, vecB), normal), np.dot(vecA, vecB))


def getAngleUnsigned(vecA:np.array, vecB:np.array) -> float:
    """Returns the magnitude of the angle between two vectors, but cannot give the sign (direction) of the angle."""
    if norm(vecA) == 0 or norm(vecB) == 0:
        raise ValueError
    
    return atan2(norm(np.cross(vecA, vecB)), np.dot(vecA, vecB))


def projectVector(vecA:np.ndarray, vecB:np.ndarray, normal:bool) -> np.ndarray:
    """Projects vector A along vector B, returns either 'parallel' or 'normal' component of vector A to vector B."""
    unitB = unit(vecB)
    parallel = np.dot(vecA, unitB)*unitB

    if normal == False:
        return parallel
    elif normal == True:
        return vecA - parallel
    else:
        raise ValueError


def grassmann(qA:np.ndarray, qB:np.ndarray) -> np.ndarray:
    """Returns the inner product of two quaternions (similar to dot-product but with orthogonal vectors)"""
    return np.r_[qA[0]*qB[0] - np.dot(qA[1:],qB[1:]), qA[0]*qB[1:] + qA[1:]*qB[0] + np.cross(qA[1:], qB[1:])]


def qRotate(vec:np.ndarray, q:np.ndarray) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    t = np.cross(2*q[1:], vec)
    return vec + q[0]*t + np.cross(q[1:], t) 


def unit(vec:np.array) -> np.array:
    """Returns a unit vector pointed in the same direction as the original vector""" 
    if norm(vec) == 0:
        raise ValueError

    return vec / norm(vec)