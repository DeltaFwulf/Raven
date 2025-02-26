from math import pi, sqrt, sin, cos
import numpy as np
from rocket.materials import *
from motion.vectorUtil import Transform



class Primitive():

    shape = "default"


    def __init__(self, name="unnamed"): # TODO: use this for part indexing later, we could create a dictionary of named parts in each module?
        self.name = name
        self.mass = 0
        self.com = np.zeros((3), float)


    def __str__(self):
        return f"{self.name}({self.shape})"
    

    def translateReference(self, tensorIn:np.array, transVec:np.array, mass:float, com:np.array=np.zeros(3)) -> np.array: 
        """
        This function uses a generalised form of the parallel axis theorem found here: https://doi.org/10.1119/1.4994835

        Inputs: 
        - object inertia tensor, I1, with axes [x1,y1,z1] about centre R1
        - translation to get to R2, about which to find the inertia tensor, I2 with axes [x2,y2,z2] parallel to [x1,y1,z1]
        - centre of mass position in frame 1 (does not have to be [0,0,0])
        - object's mass

        Outputs:
        - I2

        I' = I + M[(R2,R2)] - 2M[(R2,C)]

        if we only know the inertia tensor about, for example, the root part, I believe that C becomes different than [0,0,0]
        and the location about which the tensor is known becomes [0,0,0]. Therefore, we can still use the relation having only
        I about some arbitrary point so long as we know the point's location relative to the centre of mass

        Usual use might be to use reference location at the centre of mass to obtain the root tensor, then to change the reference location to the root for placement in the module
        """

        def getSymmetricMatrix(a, b):

            c = np.empty((3,3), float)

            c[0,0] = a[1]*b[1] + a[2]*b[2]
            c[0,1] = -0.5 * (a[0]*b[1] + a[1]*b[0])
            c[0,2] = -0.5 * (a[0]*b[2] + a[2]*b[0])

            c[1,0] = c[0,1]
            c[1,1] = a[0]*b[0] + a[2]*b[2]
            c[1,2] = -0.5 * (a[1]*b[2] + a[2]*b[1])

            c[2,0] = c[0,2]
            c[2,1] = c[1,2]
            c[2,2] = a[0]*b[0] + a[1]*b[1]

            return c


        I2 = tensorIn + (mass * getSymmetricMatrix(transVec, transVec)) - (2 * mass * getSymmetricMatrix(transVec,com))

        return I2


    def rotateReference(self, tensorIn:np.array, transform:Transform) -> np.array:

            i = np.array([1, 0, 0])
            j = np.array([0, 1, 0])
            k = np.array([0, 0, 1])

            iNew = transform.align(i)
            jNew = transform.align(j)
            kNew = transform.align(k)

            def cosAng(vec1, vec2):
                dot = np.dot(vec1, vec2)
                mag = sqrt(np.sum(vec1**2 + vec2**2))
                return dot/mag

            T = np.empty((3,3), float)
        
            T[0,0] = cosAng(iNew, i)
            T[0,1] = cosAng(iNew, j)
            T[0,2] = cosAng(iNew, k)

            T[1,0] = cosAng(jNew, i)
            T[1,1] = cosAng(jNew, j)
            T[1,2] = cosAng(jNew, k)

            T[2,0] = cosAng(kNew, i)
            T[2,1] = cosAng(kNew, j)
            T[2,2] = cosAng(kNew, k)

            return np.matmul(T, np.matmul(tensorIn, np.transpose(T)))
    

    def moveReference(self, tensorIn:np.array, comTransform:np.array, mass:float, relTransform:Transform) -> np.array:
        """
        This function calculates the inertia tensor of an object from a new reference frame given the 
        transformation between the original and new reference frames.

        Inputs:
        - untransformed tensor
        - transform between original and new reference frames
        - untransformed translation vector to the part centre of mass
        - mass of the object

        Step 1: align the mass inertia tensor to the reference axis system
        Step 2: using parallel axis theorem, add a correction from translation to the inertia tensor
        Step 3: return the inertia tensor, I2
        """

        rotating = (relTransform.getRotMatrix() != np.identity(3)).any()
        translating = (relTransform.getTransVec() != np.zeros((3), float)).any()

        tensor = tensorIn

        if rotating:
            tensor = self.rotateReference(tensorIn=tensor, 
                                          transform=relTransform)

        if translating:
            tensor = self.translateReference(tensorIn=tensor, 
                                             transVec=relTransform.getTransVec(), 
                                             mass=mass, 
                                             com=comTransform.getTransVec())
                                             
        return tensor



class Conic(Primitive):

    shape = "conic"

    def __init__(self, moduleTransform:Transform, length, dOuterRoot, dOuterEnd, dInnerRoot=0, dInnerEnd=0, name="conic", material=Material):

        self.name = name
        self.material = material
        self.density = material.density

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

        self.rootTransform = Transform(transInit=self.com)
        self.moduleTransform = moduleTransform
        self.moduleTransform.chain(self.rootTransform) # module transforms are based on the primitive root frame
   
        self.transforms = {'root':self.rootTransform, 'module':moduleTransform} # all transforms are relative to the part's reference axes (here it is the object's principal axes)
        self.comRef = Transform() # this primitive has its reference frame at the centre of mass

        # Inertia tensors
        self.moi_ref = self.calcMassTensor()
        self.moi_com = self.moi_ref
        self.moi_root = self.moveReference(tensorIn=self.moi_com, comTransform=self.comRef, mass=self.mass, relTransform=self.rootTransform)

        self.vertices, self.edges = self.wireframe()

    
    def calcMass(self):
        return (pi * self.material.density * self.length / 3) * ((self.rOuterRoot**2 - self.rInnerRoot**2) + (self.rOuterRoot * self.rOuterEnd - self.rInnerRoot * self.rInnerEnd) + (self.rOuterEnd**2 - self.rInnerEnd**2))


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
        innerTensor = getSolidTensor(self.rInnerRoot, self.rInnerEnd, self.length, self.com, self.material.density)
        outerTensor = getSolidTensor(self.rOuterRoot, self.rOuterEnd, self.length, self.com, self.material.density)

        return outerTensor - innerTensor
    

    def wireframe(self, reference:str='root'):
        """
        Returns the correct vertices and edge connections to draw the specified conic shape.
        
        Vertex order: outer ring 0, outer ring 1, inner ring 0, inner ring 1

        Rules:
        either r0_outer or r1_outer MUST be > 0 (no trivial shape)

        r0_inner must be less than r0_outer, r1_inner must be less than r1_outer

        if these rules are violated, you will be YELLED AT >:(
        """

        # TODO: build in invalid shape handling (give some dumb shape to show they messed up like a 3 sided prism of 1 x 1)

        nFace = 24 # resolution of the part
        transform = self.transforms[reference]

        # from definiition of 'root location'
        x0 = 0
        xf = x0 + self.length

        # prevent negative values
        r0_inner = max(0, self.rInnerRoot)
        r1_inner = max(0, self.rInnerEnd)
        r0_outer = max(0, self.rOuterRoot)
        r1_outer = max(0, self.rOuterEnd)

        v0_outer = []
        v0_inner = []

        v1_outer = []
        v1_inner = []

        offOut1 = nFace if r0_outer > 0 else 1
        offIn0 = offOut1 + (nFace if r1_outer > 0 else 1)
        offIn1 = offIn0 + (nFace if r0_inner > 0 else 1)

        vertices = []
        edges = []

        ang = np.linspace(0, 2*pi * (nFace - 1) / nFace, nFace)

        # create outer conic vertices
        if(r0_outer > 0 and r1_outer > 0):

            for i in range(0, nFace):

                v0_outer.append(transform.local2parent((x0, r0_outer * cos(ang[i]), r0_outer * sin(ang[i]))))
                v1_outer.append(transform.local2parent((xf, r1_outer * cos(ang[i]), r1_outer * sin(ang[i]))))

                edges.append((i, (i+1) % nFace)) # ring 0
                edges.append((i + nFace, (i + 1) % nFace + offOut1)) # ring 1
                edges.append((i, i + nFace)) # ring 0 to ring 1

        elif(r0_outer > 0):

            v1_outer.append(transform.local2parent((xf, 0, 0)))

            for i in range(0, nFace):
                
                v0_outer.append(transform.local2parent((x0, r0_outer * cos(ang[i]), r0_outer * sin(ang[i]))))

                edges.append((i, (i+1) % nFace)) # ring 0
                edges.append((i, offOut1)) # conic point

        elif(r1_outer > 0):

            v0_outer.append(transform.local2parent((x0, 0, 0)))

            for i in range(0, nFace):

                v1_outer.append(transform.local2parent((xf, r1_outer * cos(ang[i]), r1_outer * sin(ang[i]))))

                edges.append((i + offOut1, (i+1) % nFace + offOut1))
                edges.append((i + offOut1, 0))

        vertices.extend(v0_outer)
        vertices.extend(v1_outer)

        # we have 4 configurations of the inner surface:
        # 1): there is no inner hole (inners both equal 0) (no inner vertices)
        # 2): lower is 0, upper is non-zero (many to point)
        # 3): upper is 0, lower is non-zero (many to point)
        # 4): both are non-zero (many to many)

        if(r0_inner > 0 and r1_inner > 0):
            for i in range(0, nFace):
                
                v0_inner.append(transform.local2parent((x0, r0_inner * cos(ang[i]), r0_inner * sin(ang[i]))))
                v1_inner.append(transform.local2parent((xf, r1_inner * cos(ang[i]), r1_inner * sin(ang[i]))))

                edges.append((i + offIn0, (i+1) % nFace + offIn0)) # inner ring 0
                edges.append((i + offIn1, (i+1) % nFace + offIn1)) # inner ring 1
                edges.append((i + offIn0, i + offIn1)) # inner ring 0 -> inner ring 1

            vertices.extend(v0_inner)
            vertices.extend(v1_inner)
                

        elif(r0_inner == 0 and r1_inner > 0):
            
            v0_inner.append(transform.local2parent((x0, 0, 0)))

            for i in range(0, nFace):
                v1_inner.append(transform.local2parent((xf, r1_inner * cos(ang[i]), r1_inner * sin(ang[i]))))
                edges.append((i + offIn1, (i+1) % nFace + offIn1)) # inner ring 1
                edges.append((offIn0, i + offIn1)) # conic point

            vertices.extend(v0_inner)
            vertices.extend(v1_inner)

        elif(r0_inner > 0 and r1_inner == 0):

            v1_inner.append(transform.local2parent((xf, 0, 0)))

            for i in range(0, nFace):
                v0_inner.append(transform.local2parent((x0, r0_inner * cos(ang[i]), r0_inner * sin(ang[i]))))
                edges.append((i + offIn0, (i+1) % nFace + offIn0)) # inner ring 0
                edges.append((i + offIn0, offIn1)) # conic point

            vertices.extend(v0_inner)
            vertices.extend(v1_inner)

        return vertices, edges
        


class RectangularPrism(Primitive):

    shape = "rectangular_prism"

    def __init__(self, x, y, z, transform:Transform, name="unnamed", material=Material):

        self.name = name
        self.transform = transform

        self.x = x
        self.y = y
        self.z = z

        self.material = material
        
        self.mass = self.material.density * x * y * z
        self.com = 0.5 * np.array((x, y, z), float) # centre of the part

        # Inertia tensors
        self.moi_com = self.calcMassTensor() # inertia tensor about centre of mass
        self.moi_root = self.translateReference(self.moi_com, self.com) # interia tensor about part root location
        self.moi_ref = self.moveReference(self.moi_com, self.transform, reference='root') # inertia tensor about module root

        self.vertices, self.edges = self.wireframe()


    def calcMass(self):
        return self.material.density * self.x * self.y * self.z
    

    def calcCoM(self):
        
        CoM = np.zeros((3), float)
        CoM[0] = self.x / 2
        return CoM
    

    def calcMassTensor(self):

        MoI = np.zeros((3,3), float)

        #TODO: input inertia tensor calculations

        return MoI
    

    def wireframe(self):

        transform = self.transform

        # first face offset:
        x0 = 0
        xf = x0 + self.x
        y0 = -0.5 * self.y
        yf = 0.5 * self.y
        z0 = -0.5 * self.z
        zf = 0.5 * self.z

        #cube vertices:
        vertices = (transform.local2parent((x0, y0, z0)),
                    transform.local2parent((x0, y0, zf)),
                    transform.local2parent((x0, yf, z0)),
                    transform.local2parent((x0, yf, zf)),
                    transform.local2parent((xf, y0, z0)),
                    transform.local2parent((xf, y0, zf)),
                    transform.local2parent((xf, yf, z0)),
                    transform.local2parent((xf, yf, zf)))

        edges = ((0,1),
                (0,2),
                (0,4),
                (1,3),
                (1,5),
                (2,3),
                (2,6),
                (3,7),
                (4,5),
                (4,6),
                (5,7),
                (6,7))

        return vertices, edges
    


class customShape(Primitive):

    shape = "custom"

    def __init__(self, mass, CoM, MoI, name="unnamed", material=Material):

        self.name = name
        self.material = material

        self.mass = mass
        self.CoM = CoM
        self.MoI_centre = MoI # at principal axes (CoM)