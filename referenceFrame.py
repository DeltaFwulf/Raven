import numpy as np
from math import sin, cos

from vectorUtil import rotateQuaternion, grassmann, unit, projectVector, getAngleSigned

class ReferenceFrame():
    """This class represents the transformation between two reference frames. Use this class to map vectors or points between different axis systems.
       Use this class to create reference frames, move reference frames, or map vectors into different frames of reference."""

    def __init__(self, rotationCfg:dict=None, origin:np.array=np.zeros(3, float)):

        self.origin = origin

        if rotationCfg is None:
            self.q = np.array([1,0,0,0], float)
            return

        if rotationCfg.get('mode') == 'axis-angle':
            axis = rotationCfg.get('axis')
            ang = rotationCfg.get('angle')
            self.q = np.r_[cos(ang / 2), sin(ang / 2)*axis]

        elif rotationCfg.get('mode') == 'vectors':
            seq = rotationCfg.get('sequence')
            locs = rotationCfg.get('vectors')

            if len(locs) > 1:
                locs[1] = projectVector(locs[1], locs[0], 'normal')

            if np.linalg.norm(locs[1]) == 0:
                raise ValueError # both vectors cannot be parallel

            parents = []
            baseVecs = {'x':np.array([1,0,0], float), 'y':np.array([0,1,0], float), 'z':np.array([0,0,1], float)}
            for vec in seq[:2]:
                parents.append(baseVecs[vec])
                
            self.q = np.array([1,0,0,0], float)
            for i in range(len(parents)):
                axis = unit(np.cross(parents[i], locs[i]))
                ang = getAngleSigned(parents[i], locs[i], axis) 
                self.q = grassmann(self.q, np.r_[cos(ang / 2), sin(ang / 2)*axis])


    def moveFrame(self, origin:np.array=np.zeros(3, float), axis:np.array=None, ang:float=None, trans:np.array=None, originFrame:str='local', moveFrame:str='parent') -> None:
        """This function moves the reference frame about a specified origin by a rotation and translation. The origin may be expressed in local or parent
           frames, as can the movement working frame."""
        
        if originFrame == 'local':
            origin = self.local2parent(origin, incTranslation=True)

        qr = np.array([1,0,0,0], float)

        if axis is not None:
            if moveFrame == 'local':
                axis = self.local2parent(axis, incTranslation=False)

            qr = np.r_[cos(ang / 2), sin(ang / 2)*unit(axis)]
            self.q = unit(grassmann(qr, self.q))
            self.origin = origin + rotateQuaternion(self.origin - origin, qr)

        if trans is not None:
            self.origin += self.local2parent(trans, incTranslation=False) if moveFrame == 'local' else trans            

    def local2parent(self, xLoc:np.array, incTranslation:bool=True) -> np.array:
        """Given a vector expressed within this reference frame, expresses the vector in the parent frame's coordinate system. If incTranslation == False,
           the vector only undergoes rotation (as though this frame shares an origin with the parent frame)."""
        
        xParent = rotateQuaternion(xLoc, self.q)
        
        if incTranslation:
            xParent += self.origin

        return xParent
        

    def parent2local(self, xParent:np.array, incTranslation:bool=True) -> np.array:
        """Given a vector expressed within the parent frame, expresses the vector in this frame's coordinate system. If incTranslation == False,
           the vector only undergoes rotation (as though this frame shares an origin with the parent frame)."""

        if incTranslation:
            xParent -= self.origin

        return rotateQuaternion(xParent, np.r_[self.q[0], -1*self.q[1:]])
    

    def invert(self) -> None:
        """Inverts the transformation such that if a vector was mapped into frame A and then A^(-1) or vice versa, the original vector would be returned"""
        self.q[1:] *= -1.0
        self.origin *= -1.0

    
    def chain(self, nextFrame:'ReferenceFrame') -> None:
            """This combines two subsequent transformations together, in the order of self.transform, newTransform"""
            self.q = grassmann(self.q, nextFrame.q)
            self.origin += nextFrame.origin