from math import pi
import numpy as np

from materials import *
from rigidBody import RigidBody



class Primitive(RigidBody):
    
    def __init__(self, **kwargs): # TODO: use this for part indexing later, we could create a dictionary of named parts in each module?
        self.name = 'unnamed' if kwargs.get('name') is None else kwargs.get('name')
        self.mass = 0.0
        self.com = np.zeros((3), float)
        self.moi = np.zeros((3,3), float)
        self.pts, self.tris = self.getMesh()

    
    def getMesh(self) -> tuple['np.ndarray', 'list']:
        pass



class Conic(Primitive):
    """This primitive represents a generalised conic frustum shape, with straight edges and two parallel ends. The shape is defined by 5 parameters: the length,
       the root outer diameter, the end outer diameter, the root inner diameter, and the end inner diameter. The shape is constrained such that up to one outer diameter
       may be zero, and that the inner diameters cannot exceed the outer diameters, but they may be equal."""

    def __init__(self, density:float, length:float, dOuterRoot:float, dOuterEnd:float, **kwargs):

        if length <= 0 or dOuterRoot < 0 or dOuterEnd < 0 or density <= 0 or (dOuterRoot == 0 and dOuterEnd == 0):
            raise ValueError
        
        self.name = 'unnamed' if kwargs.get('name') is None else kwargs.get('name')
        self.density = density
        self.length = length
        self.rOuterRoot = dOuterRoot / 2
        self.rOuterEnd = dOuterEnd / 2
        self.rInnerRoot = 0 if kwargs.get('dInnerRoot') is None else kwargs.get('dInnerRoot') / 2
        self.rInnerEnd = 0 if kwargs.get('dInnerEnd') is None else kwargs.get('dInnerEnd') / 2
 
        if self.rInnerRoot < 0 or self.rInnerEnd < 0 or self.rInnerRoot > self.rOuterRoot or self.rInnerEnd > self.rOuterEnd:
            raise ValueError

        self.mass, self.com, self.moi = self.getInertialParams()
        self.pts, self.tris = self.getMesh()

    
    def getInertialParams(self) -> tuple['float', 'np.ndarray', 'np.ndarray']:
        """Calculates the mass, centre of mass vector, and moment of inertia tensor of the conic shape
           This isn't as direct a calculation as the integral method, but is computationally more efficient"""
        

        def calcSolid(rho:float, h:float, rRoot:float, rEnd:float) -> tuple['float', 'float', 'float', 'float']:

            if rRoot == rEnd:
                m = rho*h*pi*rRoot**2
                x = -h / 2
                Ix = m*rRoot**2 / 2
                Ip = m*((3*rRoot**2 + h**2) / 12 + x**2)

            else:
                r0 = min(rRoot, rEnd)
                r1 = max(rRoot, rEnd)
                rootpt = r0 == rRoot

                hb = h*(r1 / (r1 - r0))
                hl = hb - h
                mb = rho*pi*r1**2 * hb / 3
                ml = rho*pi*r0**2 * hl / 3
                xb = hl - 0.75*hb if rootpt else -hb / 4
                xl = hl / 4 if rootpt else 0.75*hl - hb

                m = mb - ml
                x = (mb*xb - ml*xl) / m
                Ix = 0.3*(mb*r1**2 - ml*r0**2)
                Ip = mb*(0.0375*hb**2 + 0.15*r1**2 + xb**2) - ml*(0.0375*hl**2 + 0.15*r0**2 + xl**2)

            return m, x, Ix, Ip
        

        if self.length == 0:
            return 0, np.zeros(3, float), np.zeros((3,3), float) # TODO: decide whether to raise an error here or return this stupid answer

        mo, xo, Ix_o, Ip_o = calcSolid(self.density, self.length, self.rOuterRoot, self.rOuterEnd)

        if self.rInnerRoot == 0 and self.rInnerEnd == 0:
            
            I = np.zeros((3, 3), float)
            I[0, 0] = Ix_o
            I[1, 1] = Ip_o - mo*xo**2
            I[2, 2] = I[1, 1]
            return mo, np.array([xo, 0, 0], float), I

        mi, xi, Ix_i, Ip_i = calcSolid(self.density, self.length, self.rInnerRoot, self.rInnerEnd)
        m = mo - mi        
        x = (mo*xo - mi*xi) / (mo - mi)
        I = np.zeros((3, 3), float)
        I[0, 0] = Ix_o - Ix_i
        I[1, 1] = Ip_o - Ip_i - m*x**2
        I[2, 2] = I[1, 1]

        return m, np.array([x, 0, 0], float), I
        

    def getMesh(self):

        """Returns points and triangles to be used in mayavi's triangular_mesh function"""
        n = 20
        theta = np.linspace(0, 2*pi, n, endpoint=False)

        rOutRoot = self.rOuterRoot
        rOutEnd = self.rOuterEnd

        rInRoot = self.rInnerRoot
        rInEnd = self.rInnerEnd

        l = self.length

        # Outer wall points
        yOutRoot = rOutRoot*np.cos(theta) if rOutRoot > 0 else np.array([0], float)
        yOutEnd = rOutEnd*np.cos(theta) if rOutEnd > 0 else np.array([0], float)
        
        zOutRoot = rOutRoot*np.sin(theta) if rOutRoot > 0 else np.array([0], float)
        zOutEnd = rOutEnd*np.sin(theta) if rOutEnd > 0 else np.array([0], float)

        xOutRoot = np.zeros_like(yOutRoot)
        xOutEnd = -l*np.ones_like(yOutEnd)

        # Inner wall points
        yInRoot = rInRoot*np.cos(theta) if rInRoot > 0 else np.array([0], float)
        yInEnd = rInEnd*np.cos(theta) if rInEnd > 0 else np.array([0], float)

        zInRoot = rInRoot*np.sin(theta) if rInRoot > 0 else np.array([0], float)
        zInEnd = rInEnd*np.sin(theta) if rInEnd > 0 else np.array([0], float)

        xInRoot = np.zeros_like(yInRoot)
        xInEnd = -l*np.ones_like(yInEnd)

        x = np.r_[xOutRoot, xOutEnd, xInRoot, xInEnd]
        y = np.r_[yOutRoot, yOutEnd, yInRoot, yInEnd]
        z = np.r_[zOutRoot, zOutEnd, zInRoot, zInEnd]

        pts = np.zeros((x.size, 3), float)
        pts[:,0] = x
        pts[:,1] = y
        pts[:,2] = z

        nro = xOutRoot.size
        neo = xOutEnd.size

        tris = []

        if nro == 1 and neo > 1:
            for i in range(0, neo):
                tris.append ((0, nro + i, nro + (i + 1)%neo))

        elif nro > 1 and neo == 1:
            for i in range(0, nro):
                tris.append((nro, i, (i + 1)%nro))

        elif nro > 1 and neo > 1:
            for i in range(0, nro):
                tris.append((i, (i + 1)%nro, nro + i))
                tris.append(((i + 1)%nro, nro + i, nro + (i + 1)%nro))

        else:
            print(f"{self.name}: invalid shape provided, cancelling")
            return

        po = xOutRoot.size + xOutEnd.size

        # inner wall:
        nri = xInRoot.size
        nei = xInEnd.size
    
        if nri == 1 and nei > 1:
            for i in range(0, nei):
                tris.append((po, po + nri + i, po + nri + (i + 1)%nei))

        elif nri > 1 and nei == 1:
            for i in range(0, nri):
                tris.append((po + nri, po + i, po + (i + 1)%nri))

        elif nri > 1 and nei > 1:
            for i in range(0, nri):
                tris.append((po + i, po + (i + 1)%nri, po + nri + i))
                tris.append((po + (i + 1)%nri, po + nri + i, po + nri + (i + 1)%nri))

        # ROOT AND END FACES ##########################################################################################################
        # if either rOutRoot or rOutEnd == 0, corresponding inner size is also == 0, so no stitching is required

        if rOutRoot > 0 and rInRoot == 0:
            for i in range(0, nro):
                tris.append((po, i, (i + 1)%nro))

        elif (rOutRoot > 0 and rInRoot > 0) and (rOutRoot != rInRoot):
            for i in range(0, nro):
                tris.append((i, (i + 1) % nro, po + i))
                tris.append(((i + 1)%nro, po + i, po + (i + 1)%nro))

        if rOutEnd > 0 and rInEnd == 0:
            for i in range(0, neo):
                tris.append((nro, nro + i, nro + (i + 1)%neo))

        elif (rOutEnd > 0 and rInEnd > 0) and (rOutEnd != rInEnd):
            for i in range(0, neo):
                tris.append((nro + i, nro + (i + 1)%neo, po + nri + i))
                tris.append((nro + (i + 1)%neo, nri + po + i, nri + po + (i + 1)%nei))

        return pts, tris
    

    # def getMass(self) -> float:
    #     return (pi * self.density * self.length / 3) * ((self.rOuterRoot**2 - self.rInnerRoot**2) + (self.rOuterRoot * self.rOuterEnd - self.rInnerRoot * self.rInnerEnd) + (self.rOuterEnd**2 - self.rInnerEnd**2))


    # def getCoM(self) -> np.ndarray:

    #     CoM = np.zeros((3), float)
    #     CoM[0] = -(self.length / 4) * ((self.rOuterRoot**2 - self.rInnerRoot**2) + 2 * (self.rOuterRoot * self.rOuterEnd - self.rInnerRoot * self.rInnerEnd) + 3 * (self.rOuterEnd**2 - self.rInnerEnd**2)) / \
    #              ((self.rOuterRoot**2 - self.rInnerRoot**2) + (self.rOuterRoot * self.rOuterEnd - self.rInnerRoot * self.rInnerEnd) + (self.rOuterEnd**2 - self.rInnerEnd**2))
        
    #     return CoM
    

    # def getInertiaTensor(self) -> np.ndarray:
        

    #     def getSolidTensor(rRoot, rEnd, l, com, density): # Calculate moment of inertia tensor about principal axes aligned to root frame

    #         I = np.zeros((3,3), float)

    #         if(rRoot != rEnd):
    #             I[0,0] = (pi * density * l / 10) * (rEnd**5 - rRoot**5) / (rEnd - rRoot)    
    #         else:
    #             I[0,0] = (pi * density * l / 2) * rRoot**4
                
    #         # Iyy, Izz:
    #         xr = com[0]
    #         xe = com[0] + l

    #         dr = rEnd - rRoot
    #         k = dr / l if l > 0 else 0

    #         A = ((k**2 / 5) * (xe**5 - xr**5) + (k/2) * (rRoot - k*xr) * (xe**4 - xr**4) + (rRoot - k*xr)**2 * (xe**3 - xr**3) / 3)
    #         B = (l / 20) * ((5 * rRoot**4) + (10 * dr * rRoot**3) + (10 * dr**2 * rRoot**2) + (5 * dr**3 * rRoot) + (dr**4))

    #         I[1,1] = pi*density*(A + B) # this has been verified for cylindrical case as well, with errors on the order of 1e-13
    #         I[2,2] = I[1,1]

    #         return I
        

    #     innerTensor = getSolidTensor(self.rInnerRoot, self.rInnerEnd, self.length, self.com, self.density)
    #     outerTensor = getSolidTensor(self.rOuterRoot, self.rOuterEnd, self.length, self.com, self.density)

    #     return outerTensor - innerTensor



class RectangularPrism(Primitive):

    def __init__(self, density:float, x:float, y:float, z:float, **kwargs) -> None:

        self.name = 'unnamed' if kwargs.get('name') is None else kwargs.get('name')
        self.mass = float(density*x*y*z)

        if self.mass <= 0:
            raise ValueError
        
        self.com = np.array([-x / 2, 0, 0], float)

        self.moi = np.zeros((3, 3), float)
        self.moi[0,0] = self.mass / 12 * (y**2 + z**2)
        self.moi[1,1] = self.mass / 12 * (x**2 + z**2)
        self.moi[2,2] = self.mass / 12 * (x**2 + y**2)

        self.pts, self.tris = self.getMeshData(x, y, z)


    def getMeshData(self, x:float, y:float, z:float) -> tuple['np.ndarray', 'list']:
        pts = np.zeros((8, 3), float)
        pts[:,0] = x*np.array([0, -1, -1, 0, 0, -1, -1, 0], float)
        pts[:,1] = y*np.array([-0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, 0.5], float)
        pts[:,2] = z*np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5], float)

        tris = [(0, 1, 4), (1, 4, 5),
                (1, 2, 5), (2, 5, 6),
                (2, 3, 6), (3, 6, 7),
                (3, 0, 7), (0, 7, 4),
                (0, 1, 2), (0, 2, 3),
                (4, 5, 6), (4, 6, 7)]

        return pts, tris
    


class TriangularPrism(Primitive):

    # The root of this part is the first point passed into this part, at x = 0

    def __init__(self, density:float, thickness:float, pts:np.ndarray, **kwargs) -> None:
        """The 2d array pts specifies, by row, the y and z values in the trianglular profile."""
        
        self.name = 'unnamed' if kwargs.get('name') is None else kwargs.get('name')
        self.mass = density*thickness*0.5*abs((pts[0,0] - pts[0,2])*(pts[1, 1] - pts[1, 0]) - (pts[0, 0] - pts[0, 1])*(pts[1, 2] - pts[1, 0]))
        self.com = np.r_[-thickness / 2, np.reshape(np.sum(pts, 1) / 3, 3)]
        self.moi = np.zeros((3, 3), float) # about centre of mass

        if np.unique(pts[0,:], return_index=True).size < 3: # The triangle must be split

            # get the middle point wrt y

            # generate shared point

            # generate both new triangles

            # solve moi of each triangle

            # use parallel axis theorem to reconstruct total triangular prism moi

            pass