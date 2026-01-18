import numpy as np
from math import sin, cos, pi

from vectorUtil import qRotate, grassmann, unit, projectVector, getAngleSigned



class ReferenceFrame():
    """This class represents the transformation between two reference frames. Use this class to map vectors or points between different axis systems.
       Use this class to create reference frames, move reference frames, or map vectors into different frames of reference."""

    def __init__(self):
        self.q = np.array([1,0,0,0], float)
        self.origin = np.zeros(3, float)
    

    def placeAxisAngle(self, axis:np.ndarray, ang:float, origin:np.ndarray=np.zeros(3, float)) -> None:
        """Used to place the reference frame within the parent frame. The frame's origin is supplied
           and it is rotated by ang about the axis at this frame's origin"""

        self.origin = origin

        axis = unit(axis)
        ang = ang % (2*pi)
        self.q = np.r_[cos(ang / 2), sin(ang / 2)*axis]

    
    def placeBaseVectors(self, seq:list['str'], vectors:list['np.ndarray'], origin:np.ndarray=np.zeros(3, float)) -> None:
        """A sequence of one or two named base vectors are used to define this frame's orientation.
           If the second vector is not supplied normal to the first, it is projected normal"""
        
        self.origin = origin
        baseVecs = {'x':np.array([1,0,0], float), 'y':np.array([0,1,0], float), 'z':np.array([0,0,1], float)}
        axis = unit(np.cross(baseVecs[seq[0]], vectors[0]))
        ang = getAngleSigned(baseVecs[seq[0]], vectors[0], axis)
        self.q = grassmann(np.r_[cos(ang / 2), sin(ang / 2)*axis], self.q)

        if len(seq) > 1:
            vectors[1] = projectVector(vectors[1], vectors[0], normal=True)

            if np.any(np.linalg.norm(vectors) == 0): # cannot have zero length vectors or parallel
                raise ValueError
            elif len(seq) > len(vectors):
                raise ValueError # TODO: provide a useful note for why

            axis = qRotate(baseVecs[seq[0]], self.q)
            target = qRotate(baseVecs[seq[1]], self.q)
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