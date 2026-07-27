"""Utility functions for 3D vectors."""

from math import pi, cos, sin, atan2, isnan
import numpy as np
from numpy.linalg import norm



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


def angle_signed(a:np.ndarray, b:np.ndarray, normal:np.ndarray) -> float:
    """Gets the angle from vecA to vecB in the correct direction, given that they both lie on a known plane."""

    if norm(a) == 0 or norm(b) == 0 or norm(normal) == 0:
        raise ValueError

    a /= norm(a)
    b /= norm(b)
    normal /= norm(normal)

    if norm((a + b)*normal) > 1e-12: # vectors must be coplanar and no vector may have zero-length
        raise ValueError

    return atan2(np.dot(np.cross(a, b), normal), np.dot(a, b))


def angle_unsigned(a:np.ndarray, b:np.ndarray) -> float:
    """Returns the magnitude of the angle between two vectors, but cannot give the sign (direction) of the angle."""
    if norm(a) == 0 or norm(b) == 0:
        raise ValueError

    return atan2(norm(np.cross(a, b)), np.dot(a, b))


def project_vector(a:np.ndarray, b:np.ndarray, normal:bool) -> np.ndarray:
    """Projects vector A along vector B, returns either 'parallel' or 'normal' component of vector A to vector B."""
    unit_b = unit(b)
    parallel = np.dot(a, unit_b)*unit_b

    if normal:
        return a - parallel

    return parallel


def grassmann(qa:np.ndarray, qb:np.ndarray) -> np.ndarray:
    """Returns the inner product of two quaternions (similar to dot-product but with orthogonal vectors)"""
    return np.r_[qa[0]*qb[0] - np.dot(qa[1:],qb[1:]), qa[0]*qb[1:] + qa[1:]*qb[0] + np.cross(qa[1:], qb[1:])]


def rotate_quaternion(vec:np.ndarray, q:np.ndarray) -> np.ndarray:
    """Rotates a vector given a unit quaternion.
    
    Source of equation: 
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles"""

    t = np.cross(2*q[1:], vec)
    return vec + q[0]*t + np.cross(q[1:], t)


def rotate_axis_angle(vec:np.ndarray, axis:np.ndarray, ang:float) -> np.ndarray:
    """Rotates the input vector by ang about the axis of rotation."""

    q = np.r_[cos(ang / 2), unit(axis)*sin(ang / 2)]
    return rotate_quaternion(vec, q)


def unit(vec:np.array) -> np.array:
    """Returns a unit vector pointed in the same direction as the original vector""" 
    if norm(vec) == 0:
        raise ValueError

    return vec / norm(vec)
