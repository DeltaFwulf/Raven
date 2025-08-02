import mayavi.mlab as maya
import numpy as np
from math import pi

from rocket.primitives import Conic, RectangularPrism
from rocket.materials import Aluminium
from utility.vectorUtil import ReferenceFrame

def tri_mesh():

    # Draw a 3D Object using mayavi triangle mesh feature

    # # let's draw a cube:
    # l = 1.0

    # points = np.zeros((8,3), float)

    # points[:, 0] = [0, 1, 1, 0, 0, 1, 1, 0]
    # points[:, 1] = [0, 0, 1, 1, 0, 0, 1, 1]
    # points[:, 2] = [0, 0, 0, 0, 1, 1, 1, 1]

    # points *= l

    # triangles = [(0, 1, 4), (1, 4, 5),
    #              (1, 2, 5), (2, 5, 6),
    #              (2, 3, 6), (3, 6, 7),
    #              (3, 0, 7), (0, 7, 4)]
    #             #  (0, 1, 2), (0, 2, 3),
    #             #  (4, 5, 6), (4, 6, 7)]
    

    # now, let's draw a cylinder
    l = 1
    rOutRoot = 0.5
    rOutEnd = 0.2
    rInRoot = 0.1
    rInEnd = 0.1
    
    n = 20

    theta = np.linspace(0, 2*pi, n, endpoint=False)

    # Outer wall points
    yOutRoot = rOutRoot*np.cos(theta) if rOutRoot > 0 else np.array([0], float)
    yOutEnd = rOutEnd*np.cos(theta) if rOutEnd > 0 else np.array([0], float)
    
    zOutRoot = rOutRoot*np.sin(theta) if rOutRoot > 0 else np.array([0], float)
    zOutEnd = rOutEnd*np.sin(theta) if rOutEnd > 0 else np.array([0], float)

    xOutRoot = np.zeros_like(yOutRoot)
    xOutEnd = l*np.ones_like(yOutEnd)

    # Inner wall points
    yInRoot = rInRoot*np.cos(theta) if rInRoot > 0 else np.array([0], float)
    yInEnd = rInEnd*np.cos(theta) if rInEnd > 0 else np.array([0], float)

    zInRoot = rInRoot*np.sin(theta) if rInRoot > 0 else np.array([0], float)
    zInEnd = rInEnd*np.sin(theta) if rInEnd > 0 else np.array([0], float)

    xInRoot = np.zeros_like(yInRoot)
    xInEnd = l*np.ones_like(yInEnd)

    x = np.r_[xOutRoot, xOutEnd, xInRoot, xInEnd]
    y = np.r_[yOutRoot, yOutEnd, yInRoot, yInEnd]
    z = np.r_[zOutRoot, zOutEnd, zInRoot, zInEnd]

    # WALL TRIANGLES ##############################################################################################################

    # outer wall
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
        print(f"invalid shape provided, cancelling")
        exit()

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

    fig = maya.figure()
    maya.triangular_mesh(x, y, z, tris, figure=fig)
    maya.show()




def drawPrimitives():

    primitives = {}
    primitives.update({'conic':Conic(1.0, 1.0, 0.5, 0.5, 0.25, material=Aluminium)})
    primitives.update({'rect':RectangularPrism(x=1, y=1, z=1, material=Aluminium)})

    frames = {}
    frames.update({'conic':ReferenceFrame(translation=np.array([0, 0, 0], float), axis=np.array([0,1,0], float), ang=0.0)})
    frames.update({'rect':ReferenceFrame(translation=np.array([5, 0, 0], float))})

    # draw all primitives in the scene
    fig = maya.figure()

    # the function should be able to draw each of the primitives in the list, and also to permit animation / updating all of the primitives' frames

    for key in primitives:
        
        pts, tris = primitives[key].getMeshData() # all points are returned within the primitive local frame

        for i in range(0, np.shape(pts)[0]):
            pts[i,:] = frames[key].local2parent(pts[i,:])

        maya.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], tris, figure=fig)
    
    maya.show()

    # simulate angular movement of the system and animate the plot

drawPrimitives()
