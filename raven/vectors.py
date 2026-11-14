"""Utility functions for 3D vectors."""

from math import pi, cos, sin, atan2, isnan
import numpy as np
from numpy.linalg import norm
import numba



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
    parallel = np.dot(a, b)*b / norm(b)**2

    return (a - parallel) if normal else parallel


def hamilton(qa:np.ndarray, qb:np.ndarray) -> np.ndarray:
    """Multiply two quaternions together, aka the Hamilton Product."""
    return np.r_[qa[0]*qb[0] - np.dot(qa[1:],qb[1:]), qa[0]*qb[1:] + qa[1:]*qb[0] + cross(qa[1:], qb[1:])]


def rotate_quaternion(vec:np.ndarray, q:np.ndarray) -> np.ndarray:
    """Rotates a vector given a unit quaternion.
    
    Source of equation: 
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles"""

    t = cross(2*q[1:], vec)
    return vec + q[0]*t + cross(q[1:], t)


def nhamilton(qa:np.ndarray, qb:np.ndarray, invert:bool=False) -> np.ndarray:
    """Chains two sets of quaternions together, qb qa, to obtain a combined rotation.
    
    If invert is True, applied qb's inverse to qa."""

    k_inv = -1 if invert else 1

    ndot = np.sum(qa[:, 1:]*k_inv*qb[:, 1:], axis=1)
    return np.r_[qa[:, 0]*qb[:, 0] - ndot, qa[:, 0]*k_inv*qb[:, 1:] + qa[:, 1:]*qb[:, 0] + np.cross(qa[:, 1:], k_inv*qb[:, 1:], axis=1)]


def transform(vec:np.ndarray, q:np.ndarray, origin:np.ndarray, invert:bool=False) -> np.ndarray:
    """Applies frame transformation, to a 1D vector.
    
    Transforms are given as an 1x7 array: [q0, q1, q2, q3, x0, x1, x2]"""

    if invert:
        vec -= origin
        t = cross(2*-q[1:], vec)
        return vec + q[0]*t + cross(-q[1:], t)

    t = cross(2*q[1:], vec)
    return vec + q[0]*t + cross(q[1:], t) + origin


def ntransform(vecs:np.ndarray, q:np.ndarray, origin:np.ndarray, invert:bool=False) -> np.ndarray:
    """Applies frame transformation, row-wise to a set of vectors.
    
    Transforms are given as an nx7 array: [q0, q1, q2, q3, x0, x1, x2]"""

    if invert:
        vecs -= origin
        t = np.cross(2*-q[:, 1:], vecs, axis=1)
        return vecs + q[:, 0].reshape(-1, 1)*t + np.cross(-q[:, 1:], t, axis=1)

    t = np.cross(2*q[:, 1:], vecs, axis=1)
    return vecs + q[:, 0]*t + np.cross(q[:, 1:], t, axis=1) + origin


def rotate_axis_angle(vec:np.ndarray, axis:np.ndarray, ang:float) -> np.ndarray:
    """Rotates the input vector by ang about the axis of rotation."""

    q = np.r_[cos(ang / 2), axis*sin(ang / 2) / norm(axis)]
    return rotate_quaternion(vec, q)


@numba.jit(nopython=True)
def cross(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    """Pure python cross product, with lower overhead than np.cross, maybe?"""

    i = a[1]*b[2] - a[2]*b[1]
    j = a[2]*b[0] - a[0]*b[2]
    k = a[0]*b[1] - a[1]*b[0]

    return np.array([i, j, k])


@numba.jit(nopython=True)
def quaternion2rotation_matrix(q:np.ndarray) -> np.ndarray:
    """Returns a 3x3 rotation matrix from an attitude quaternion."""

    r = np.zeros((3, 3), float)
    r[0, 0] = 1 - 2*(q[2]**2 + q[3]**2)
    r[0, 1] = 2*(q[1]*q[2] - q[3]*q[0])
    r[0, 2] = 2*(q[1]*q[3] + q[2]*q[0])
    r[1, 0] = 2*(q[1]*q[2] + q[3]*q[0])
    r[1, 1] = 1 - 2*(q[1]**2 + q[3]**2)
    r[1, 2] = 2*(q[2]*q[3] - q[1]*q[0])
    r[2, 0] = 2*(q[1]*q[3] - q[2]*q[0])
    r[2, 1] = 2*(q[2]*q[3] + q[1]*q[0])
    r[2, 2] = 1 - 2*(q[1]**2 + q[2]**2)

    return r


def relative_quaternion(a:np.ndarray, b:np.ndarray) -> np.ndarray:
    """Calculates the attitude quaternion that rotates a to b.

    a and b are assumed to have equal magnitude."""

    a /= norm(a)
    b /= norm(b)

    ang = np.arccos(np.dot(a, b))
    axis = cross(a, b)

    return np.r_[np.cos(ang / 2), np.sin(ang / 2)*axis]
