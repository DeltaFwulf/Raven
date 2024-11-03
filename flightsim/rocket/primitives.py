from math import pi, sqrt
import numpy as np
from materials import legacyMaterial
from motion.frame import Frame


class Primitive():

    shape = "default"


    def __init__(self, name="unnamed"): # TODO: use this for part indexing later, we could create a dictionary of named parts in each module?
        self.name = name
        self.mass = 0
        self.com = np.zeros((3), float)


    def __str__(self):
        return f"{self.name}({self.shape})"
    

    def translateReference(self, tensorIn:np.array, translation:np.array) -> np.array: 
            
            """!!IMPORTANT: This translation MUST only be used on the mass inertia tensor, else only translations away from the centre of mass in all directions can be calculated accurately."""

            correction = np.zeros((3,3), float)

            correction[0,0] = (translation[1]**2 + translation[2]**2)
            correction[1,1] = (translation[0]**2 + translation[2]**2)
            correction[2,2] = (translation[0]**2 + translation[1]**2)

            correction[0,1] = (translation[0] * translation[1])
            correction[0,2] = (translation[0] * translation[2])
            correction[1,2] = (translation[1] * translation[2])

            correction[1,0] = correction[0,1]
            correction[2,0] = correction[0,2]
            correction[2,1] = correction[1,2]

            return (tensorIn + (self.mass * correction)) * np.array([[1, -1, 1],[-1, 1, -1], [1,-1, 1]], float) # XXX: check this pls
    

    def rotateReference(self, tensorIn:np.array, transformation:Frame) -> np.array:

            i = np.array([1, 0, 0])
            j = np.array([0, 1, 0])
            k = np.array([0, 0, 1])

            iNew = transformation.align(i)
            jNew = transformation.align(j)
            kNew = transformation.align(k)

            def dot(vec1, vec2):

                sum = np.sum(vec1 * vec2)
                mag = sqrt(np.sum(vec1**2)) * sqrt(np.sum(vec2**2))
            
                return sum/mag

            T = np.empty((3,3), float)
        
            T[0,0] = dot(iNew, i)
            T[0,1] = dot(iNew, j)
            T[0,2] = dot(iNew, k)

            T[1,0] = dot(jNew, i)
            T[1,1] = dot(jNew, j)
            T[1,2] = dot(jNew, k)

            T[2,0] = dot(kNew, i)
            T[2,1] = dot(kNew, j)
            T[2,2] = dot(kNew, k)

            return np.matmul(T, np.matmul(tensorIn, np.transpose(T)))
    

    def moveReference(self, tensorIn:np.array, transform:Frame, reference:str='com') -> np.array:
        """
        This function calculates the inertia tensor of the primitive relative to a new frame of reference. 

        - The transformation is between the shape-aligned CoM reference system and one with a known transform.
        - Two modes can be used, root or centre, which sets the transformation origin. Root is likely to be used more as it is how parts are placed.

        Step 1: align the inertia tensor to the reference axis system
        Step 2: using parallel axis theorem, add a correction from translation to the inertia tensor
        Step 3: return the modified inertia tensor
        """

        rotated = self.rotateReference(tensorIn, transform)

        if(reference == 'root'):
            translation = transform.transVector + transform.align(self.com)
        else:
            translation = transform.transVector

        return self.translateReference(rotated, translation) # step 2, translate the reference frame (this is a translation w.r.t. the parent frame, the now 'rotated' reference, conveniently)



class Conic(Primitive):

    shape = "conic"

    def __init__(self, length, transform:Frame, dOuterRoot, dOuterEnd, dInnerRoot=0, dInnerEnd=0, name="unnamed", material="generic"):

        self.name = name
        self.material = material
        self.density = legacyMaterial.densities[material]

        self.length = length

        self.dOuterRoot = dOuterRoot
        self.dOuterEnd = dOuterEnd
        self.dInnerRoot = dInnerRoot
        self.dInnerEnd = dInnerEnd

        self.rOuterRoot = dOuterRoot / 2
        self.rOuterEnd = dOuterEnd / 2
        self.rInnerRoot = dInnerRoot / 2
        self.rInnerEnd = dInnerEnd / 2

        self.mass = self.calcMass()
        self.com = self.calcCoM()
        
        # calculate the inertia tensor about the 'root' of the part
        self.transform = transform

        # Inertia tensors
        self.moi_com = self.calcMassTensor() # inertia tensor about centre of mass
        self.moi_root = self.translateReference(self.moi_com, self.com) # interia tensor about part root location
        self.moi_ref = self.moveReference(self.moi_com, self.transform, reference='root') # inertia tensor about module root

    
    def calcMass(self):
        return (pi * self.density * self.length / 3) * ((self.rOuterRoot**2 - self.rInnerRoot**2) + (self.rOuterRoot * self.rOuterEnd - self.rInnerRoot * self.rInnerEnd) + (self.rOuterEnd**2 - self.rInnerEnd**2))


    def calcCoM(self):

        CoM = np.zeros((3), float)
        CoM[0] = (self.length / 4) * ((self.rOuterRoot**2 - self.rInnerRoot**2) + 2 * (self.rOuterRoot * self.rOuterEnd - self.rInnerRoot * self.rInnerEnd) + 3 * (self.rOuterEnd**2 - self.rInnerEnd**2)) / \
                 ((self.rOuterRoot**2 - self.rInnerRoot**2) + (self.rOuterRoot * self.rOuterEnd - self.rInnerRoot * self.rInnerEnd) + (self.rOuterEnd**2 - self.rInnerEnd**2))
        
        return CoM
    

    def calcMassTensor(self):
        
        def getSolidTensor(rootRadius, endRadius, length, CoM, density): # returns the inertia tensor of the object about its centre of mass, in the part's principal axes

            tensor = np.zeros((3,3), float)

            if(rootRadius != endRadius):
                tensor[0,0] = (pi * density * length / 10) * (endRadius**5 - rootRadius**5) / (endRadius - rootRadius)    
            else:
                tensor[0,0] = (pi * density * length / 2) * rootRadius**4
                
            # Iyy, Izz:
            x0 = -CoM[0]
            xf = length - CoM[0]

            dR = endRadius - rootRadius
            k = dR / length

            A = ((k**2 / 5) * (xf**5 - x0**5) + (k/2) * (rootRadius - k*x0) * (xf**4 - x0**4) + (1/3) * (rootRadius - k*x0)**2 * (xf**3 - x0**3))
            B = (length / 20) * ((5 * rootRadius**4) + (10 * dR * rootRadius**3) + (10 * dR**2 * rootRadius**2) + (5 * dR**3 * rootRadius) + (dR**4))

            tensor[1,1] = pi * density * (A + B) # this has been verified for cylindrical case as well, with errors on the order of 1e-13
            tensor[2,2] = tensor[1,1] # Izz = Iyy due to axisymmetry

            return tensor

        # To find the inertia tensor of the shape, we simply subtract the tensor of the inner cone from that of the outer cone:
        innerTensor = getSolidTensor(self.rInnerRoot, self.rInnerEnd, self.length, self.com, self.density)
        outerTensor = getSolidTensor(self.rOuterRoot, self.rOuterEnd, self.length, self.com, self.density)

        return outerTensor - innerTensor
        


class RectangularPrism(Primitive):

    shape = "rectangular_prism"

    def __init__(self, x, y, z, name="unnamed", material="generic"):

        self.name = name

        self.x = x
        self.y = y
        self.z = z

        self.material = material
        self.density = legacyMaterial.densities[material]


    def calcMass(self):
        return self.density * self.x * self.y * self.z
    

    def calcCoM(self):
        
        CoM = np.zeros((3), float)
        CoM[0] = self.x / 2
        return CoM
    

    def calcMoI(self):

        MoI = np.zeros((3,3), float)

        #TODO: input inertia tensor calculations

        return MoI
    


class customShape(Primitive):

    shape = "custom"

    def __init__(self, mass, CoM, MoI, name="unnamed"):

        self.mass = mass
        self.CoM = CoM
        self.MoI_centre = MoI # this is wrt to the part origin

        self.name = name
        