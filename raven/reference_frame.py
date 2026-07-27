"""Reference frame specific functions and classes."""

from math import sin, cos, pi
import numpy as np

from .vectorUtil import qRotate, grassmann, unit, projectVector, getAngleSigned



class ReferenceFrame():
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
            ang = getAngleSigned(base_vectors[seq[0]], vectors[0], axis)
            self.q = grassmann(np.r_[cos(ang / 2), sin(ang / 2)*unit(axis)], self.q)

        if len(seq) > 1:
            vectors[1] = projectVector(vectors[1], vectors[0], normal=True)

            axis = qRotate(base_vectors[seq[0]], self.q)
            target = qRotate(base_vectors[seq[1]], self.q)
            ang = getAngleSigned(vectors[1], target, axis)
            self.q = grassmann(np.r_[cos(ang / 2), sin(ang / 2)*axis], self.q)


    def moveFrame(self, origin:np.array=np.zeros(3, float), axis:np.array=None, ang:float=None, trans:np.array=None, originFrame:str='local', moveFrame:str='parent') -> None:
        """This function moves the reference frame about a specified origin by a rotation and translation. The origin may be expressed in local or parent
           frame, as can the movement working frame."""    

        if axis is not None and ang != 0:
            if originFrame == 'local':
                origin = self.local2parent(origin, incTranslation=True)

            if moveFrame == 'local':
                axis = self.local2parent(axis, incTranslation=False)

            ang = ang % (2*pi)
            qr = np.r_[cos(ang / 2), sin(ang / 2)*unit(axis)]
            self.origin = origin + qRotate(self.origin - origin, qr)
            if trans is not None:
                self.origin += self.local2parent(trans, incTranslation=False) if moveFrame == 'local' else trans            

            self.q = unit(grassmann(qr, self.q))


    def local2parent(self, xLoc:np.array, incTranslation:bool=True) -> np.array:
        """Given a vector expressed within this reference frame, expresses the vector in the parent frame's coordinate system. If incTranslation == False,
           the vector only undergoes rotation (as though this frame shares an origin with the parent frame)."""

        xParent = qRotate(xLoc, self.q)
        return xParent if not incTranslation else xParent + self.origin


    def parent2local(self, xParent:np.array, incTranslation:bool=True) -> np.array:
        """Given a vector expressed within the parent frame, expresses the vector in this frame's coordinate system. If incTranslation == False,
           the vector only undergoes rotation (as though this frame shares an origin with the parent frame)."""

        if incTranslation:
            xParent -= self.origin

        return qRotate(xParent, np.r_[self.q[0], -1*self.q[1:]])
