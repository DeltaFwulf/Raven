import mayavi.mlab as mlab
import numpy as np
from math import pi

from primitives import Conic, RectangularPrism
from materials import Aluminium
from vectorUtil import ReferenceFrame



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

    fig = mlab.figure()
    mlab.triangular_mesh(x, y, z, tris, figure=fig)
    mlab.show()



def animPrimitives():

    primitives = {}
    primitives.update({'conic':Conic(5.0, 1.0, 0, 0, 0, material=Aluminium)})
    primitives.update({'rect':RectangularPrism(x=1, y=1, z=1, material=Aluminium)})

    frames = {}
    frames.update({'conic':ReferenceFrame(translation=np.array([0, 0, 0], float), axis=np.array([1,0,0], float), ang=pi/4)})
    frames.update({'rect':ReferenceFrame(translation=np.array([5, 0, 0], float), axis=np.array([1,0,0], float), ang=pi/4)})

    # draw all primitives in the scene
    fig = mlab.figure()
    meshes = {}
    # the function should be able to draw each of the primitives in the list, and also to permit animation / updating all of the primitives' frames

    for key in primitives:
        
        pts, tris = primitives[key].getMeshData() # all points are returned within the primitive local frame

        for i in range(0, np.shape(pts)[0]):
            pts[i,:] = frames[key].local2parent(pts[i,:])

        meshes.update({key:None})
        meshes[key] = mlab.triangular_mesh(pts[:,0], pts[:,1], pts[:,2], tris, figure=fig)
        # meshes[key].mlab_source.x = pts[:,0]*5
    
    # TODO: come up with more efficient way of moving points
    # - currently, we need to recreate all points and tris to move them to final position
    # - we should either store a copy of local pts data for modification OR
    # - interpolate between two frames using a difference method within referenceFrame like diff()
    # - OR we could invert the transform of the mesh points then update the frame (however this doesn't cover variable primitive geometry)
    @mlab.animate(delay=17)
    def anim(plays:int=1):

        n = 0

        while(n < plays):

            for i in range(0, 360):
                
                # spin the primitives about the(ir) z axis
                dAng = pi / 180

                for key in primitives:
                    
                    p = primitives[key]
                    f = frames[key]

                    #TODO: solve issue of non-zero origin with local rotation

                    #f.move(axis=np.array([0,0,1], float), ang=dAng, reference='parent')
                    com = p.com
                    f.moveAbout(origin=com, axis=np.array([0,0,1], float), ang=dAng, frame='local')
                    pts = f.local2parent(p.ptsLocal)
                    meshes[key].mlab_source.trait_set(x=pts[:,0], y=pts[:,1], z=pts[:,2])
                    yield

            n += 1


    anim(100)
    mlab.show()



animPrimitives()
