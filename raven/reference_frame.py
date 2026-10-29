"""Reference frame specific functions and classes."""

from math import sin, cos, pi
from copy import deepcopy
import numpy as np

from .vectors import rotate_quaternion, grassmann, unit, project_vector, angle_signed



class Frame():
    """A 3D reference frame, defined relative to its parent frame."""

    def __init__(self):
        self.q = np.array([1,0,0,0], float)
        self.origin = np.zeros(3, float)


    def place_axis_angle(self, axis:np.ndarray, ang:float, origin:np.ndarray=np.zeros(3, float)) -> None:
        """Used to place the reference frame within the parent frame. 
        
        The frame's origin is supplied, and it is rotated
        by ang about the axis at this frame's origin."""

        self.origin = origin

        axis = unit(axis)
        ang = ang % (2*pi)
        self.q = np.r_[cos(ang / 2), sin(ang / 2)*axis]


    def place_base_vectors(self, seq:list['str'], vectors:list['np.ndarray'], origin:np.ndarray=np.zeros(3, float)) -> None:
        """A sequence of one or two named base vectors are used to define this frame's orientation.
           If the second vector is not supplied normal to the first, it is projected normal"""

        if np.any(np.linalg.norm(vectors) == 0) or len(seq) != len(vectors) or len(seq) > 2:
            raise ValueError

        self.origin = origin
        base_vectors = {'x':np.array([1,0,0], float), 'y':np.array([0,1,0], float), 'z':np.array([0,0,1], float)}
        axis = np.cross(base_vectors[seq[0]], vectors[0])

        if np.linalg.norm(axis) > 0:
            ang = angle_signed(base_vectors[seq[0]], vectors[0], axis)
            self.q = grassmann(np.r_[cos(ang / 2), sin(ang / 2)*unit(axis)], self.q)

        if len(seq) > 1:
            vectors[1] = project_vector(vectors[1], vectors[0], normal=True)

            axis = rotate_quaternion(base_vectors[seq[0]], self.q)
            target = rotate_quaternion(base_vectors[seq[1]], self.q)
            ang = angle_signed(vectors[1], target, axis)
            self.q = grassmann(np.r_[cos(ang / 2), sin(ang / 2)*axis], self.q)


    def move(self, origin:np.array=np.zeros(3, float), axis:np.array=None, ang:float=None, trans:np.array=None, origin_frame:str='local', move_frame:str='parent') -> None:
        """This function moves the reference frame about a specified origin by a rotation and translation. The origin may be expressed in local or parent
           frame, as can the movement working frame."""

        if axis is not None and ang != 0:
            if origin_frame == 'local':
                origin = self.local2parent(origin, incTranslation=True)

            if move_frame == 'local':
                axis = self.local2parent(axis, incTranslation=False)

            ang = ang % (2*pi)
            qr = np.r_[cos(ang / 2), sin(ang / 2)*unit(axis)]
            self.origin = origin + rotate_quaternion(self.origin - origin, qr)
            if trans is not None:
                self.origin += self.local2parent(trans, incTranslation=False) if move_frame == 'local' else trans            

            self.q = unit(grassmann(qr, self.q))


    def local2parent(self, xLoc:np.array, incTranslation:bool=True) -> np.array:
        """Given a vector expressed within this reference frame, expresses the vector in the parent frame's coordinate system. If incTranslation == False,
           the vector only undergoes rotation (as though this frame shares an origin with the parent frame)."""

        xParent = rotate_quaternion(xLoc, self.q)
        return xParent if not incTranslation else xParent + self.origin


    def parent2local(self, xParent:np.array, incTranslation:bool=True) -> np.array:
        """Given a vector expressed within the parent frame, expresses the vector in this frame's coordinate system. If incTranslation == False,
           the vector only undergoes rotation (as though this frame shares an origin with the parent frame)."""

        if incTranslation:
            xParent -= self.origin

        return rotate_quaternion(xParent, np.r_[self.q[0], -1*self.q[1:]])


    def invert(self) -> None:
        """Inverts the frame's transformation."""

        self.q[1:] *= -1
        self.origin *= -1


def map_inertia_tensor(frame:Frame, m:float, x:np.ndarray, i_in:np.ndarray, **kwargs) -> np.ndarray:
    """Maps a moment of inertia tensor into a new coordinate system.

    The frame passed into the argument is defined relative to the current reference frame.
    
    Unless specified, it is assumed that i_in has its origin at the body's centre of mass, else
    the location relative to the com shall be passed with keyword 'ref'.

    The body's mass and centre of mass must also be given, with com in the same frame as i_in.

    The moment of inertia tensor is first rotated and then translated, using generalised parallel
    axis theorem (https://doi.org/10.1119/1.4994835)."""

    i_out = deepcopy(i_in)

    if not np.array_equal(frame.q, np.array([1, 0, 0, 0], float)):

        i = np.array([1, 0, 0], float)
        j = np.array([0, 1, 0], float)
        k = np.array([0, 0, 1], float)

        u = frame.local2parent(i, incTranslation=False)
        v = frame.local2parent(j, incTranslation=False)
        w = frame.local2parent(k, incTranslation=False)

        rotmat = np.zeros((3,3), float)

        rotmat[0,0] = np.dot(i, u)
        rotmat[0,1] = np.dot(j, u)
        rotmat[0,2] = np.dot(k, u)

        rotmat[1,0] = np.dot(i, v)
        rotmat[1,1] = np.dot(j, v)
        rotmat[1,2] = np.dot(k, v)

        rotmat[2,0] = np.dot(i, w)
        rotmat[2,1] = np.dot(j, w)
        rotmat[2,2] = np.dot(k, w)

        i_out = np.matmul(rotmat, np.matmul(i_out, np.transpose(rotmat)))

    # ref is moi origin in the original frame
    # r is displacement vector from initial reference to frame origin, in new frame
    # c is displacement vector from initial reference point to the centre of mass, in new frame
    ref = x if kwargs.get('ref') is None else kwargs.get('ref')
    r = frame.parent2local(-ref, incTranslation=True)

    if not np.array_equal(r, np.zeros(3, float)):

        def relational_matrix(a:np.ndarray, b:np.ndarray) -> np.ndarray:

            relmat = np.zeros((3,3), float)

            relmat[0,0] = a[1]*b[1] + a[2]*b[2]
            relmat[0,1] = -0.5*(a[0]*b[1] + a[1]*b[0])
            relmat[0,2] = -0.5*(a[0]*b[2] + a[2]*b[0])

            relmat[1,0] = relmat[0,1]
            relmat[1,1] = a[0]*b[0] + a[2]*b[2]
            relmat[1,2] = -0.5*(a[1]*b[2] + a[2]*b[1])

            relmat[2,0] = relmat[0,2]
            relmat[2,1] = relmat[1,2]
            relmat[2,2] = a[0]*b[0] + a[1]*b[1]

            return relmat

        c = frame.parent2local(x - ref, incTranslation=False)
        i_out += m*(relational_matrix(r, r) - 2*relational_matrix(r, c))

    return i_out
