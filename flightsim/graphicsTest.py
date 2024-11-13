"""Let's attempt to draw a wireframe shape, then build this file into a part visualiser

The objective of this code is to be able to take in a shape definition (module, primitives, etc) and a location,
and draw the shape in 3D in a window.

"""

import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from math import cos, sin, pi
from motion.frame import Frame

# create a list of vertices for a conic() shape:
# currently, it is a 1x1 right cylinder:
# for future reference, this will be run at initialisation of each primitive, then stored in each object as tuples
def buildConic(height:float, r0_outer:float, r0_inner:float, r1_outer:float, r1_inner:float, nFace:int, frame:Frame):
    """
    Returns the correct vertices and edge connections to draw the specified conic shape.
    
    Vertex order: outer ring 0, outer ring 1, inner ring 0, inner ring 1

    Rules:
    either r0_outer or r1_outer MUST be > 0 (no trivial shape)

    r0_inner must be less than r0_outer, r1_inner must be less than r1_outer

    if these rules are violated, you will be YELLED AT >:(
    """

    # TODO: build in invalid shape handling (give some dumb shape to show they messed up like a 3 sided prism of 1 x 1)

    # from definiition of 'root location'
    x0 = 0
    xf = x0 + height

    # prevent negative values
    r0_inner = max(0, r0_inner)
    r1_inner = max(0, r1_inner)
    r0_outer = max(0, r0_outer)
    r1_outer = max(0, r1_outer)

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

            v0_outer.append(frame.map((r0_outer * cos(ang[i]), r0_outer * sin(ang[i]), x0)))
            v1_outer.append(frame.map((r1_outer * cos(ang[i]), r1_outer * sin(ang[i]), xf)))

            edges.append((i, (i+1) % nFace)) # ring 0
            edges.append((i + nFace, (i + 1) % nFace + offOut1)) # ring 1
            edges.append((i, i + nFace)) # ring 0 to ring 1

    elif(r0_outer > 0):

        v1_outer.append(frame.map((0, 0, xf)))

        for i in range(0, nFace):
            
            v0_outer.append(frame.map((r0_outer * cos(ang[i]), r0_outer * sin(ang[i]), x0)))

            edges.append((i, (i+1) % nFace)) # ring 0
            edges.append((i, offOut1)) # conic point

    elif(r1_outer > 0):

        v0_outer.append(frame.map((0, 0, x0)))

        for i in range(0, nFace):

            v1_outer.append(frame.map((r1_outer * cos(ang[i]), r1_outer * sin(ang[i]), xf)))

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
            
            v0_inner.append(frame.map((r0_inner * cos(ang[i]), r0_inner * sin(ang[i]), x0)))
            v1_inner.append(frame.map((r1_inner * cos(ang[i]), r1_inner * sin(ang[i]), xf)))

            edges.append((i + offIn0, (i+1) % nFace + offIn0)) # inner ring 0
            edges.append((i + offIn1, (i+1) % nFace + offIn1)) # inner ring 1
            edges.append((i + offIn0, i + offIn1)) # inner ring 0 -> inner ring 1

        vertices.extend(v0_inner)
        vertices.extend(v1_inner)
            

    elif(r0_inner == 0 and r1_inner > 0):
        
        v0_inner.append(frame.map([0, 0, x0]))

        for i in range(0, nFace):
            v1_inner.append(frame.map((r1_inner * cos(ang[i]), r1_inner * sin(ang[i]), xf)))
            edges.append((i + offIn1, (i+1) % nFace + offIn1)) # inner ring 1
            edges.append((offIn0, i + offIn1)) # conic point

        vertices.extend(v0_inner)
        vertices.extend(v1_inner)

    elif(r0_inner > 0 and r1_inner == 0):

        v1_inner.append(frame.map([0, 0, xf]))

        for i in range(0, nFace):
            v0_inner.append(frame.map((r0_inner * cos(ang[i]), r0_inner * sin(ang[i]), x0)))
            edges.append((i + offIn0, (i+1) % nFace + offIn0)) # inner ring 0
            edges.append((i + offIn0, offIn1)) # conic point

        vertices.extend(v0_inner)
        vertices.extend(v1_inner)

    return vertices, edges



def rectPrism():

    #cube vertices:
    vertices = ((1, -1, -1),
                (1, 1, -1),
                (-1, 1, -1),
                (-1, -1, -1),
                (1, -1, 1),
                (1, 1, 1),
                (-1, -1, 1),
                (-1, 1, 1))

    edges = ((0,1),
            (0,3),
            (0,4),
            (2,1),
            (2,3),
            (2,7),
            (6,3),
            (6,4),
            (6,7),
            (5,1),
            (5,4),
            (5,7))

    return vertices, edges

def worldFrame():

    vertices = ((0, 0, 0),
                (1, 0, 0),
                (0, 1, 0),
                (0, 0, 1))

    edges = ((0,1),
             (0,2),
             (0,3))
    
    return vertices, edges





def draw(vertices:tuple, edges:tuple):

    glBegin(GL_LINES)
    
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])

    glEnd()


def main():

    vertFrame, edgeFrame = worldFrame()

    transform0 = Frame(np.array([0,0,0], float), angInit=0, axisInit=np.array([1,0,0], float))
    transform1 = Frame(np.array([0,0,1], float), angInit=0, axisInit=np.array([1,0,0], float))
    transform2 = Frame(np.array([0,0,2], float), angInit=0, axisInit=np.array([1,0,0], float))

    vertices0, edges0 = buildConic(height=1, r0_outer=0.5, r0_inner=0, r1_outer=0, r1_inner=0, nFace=24, frame=transform0)
    vertices1, edges1 = buildConic(height=1, r0_outer=0.5, r0_inner=0, r1_outer=0.5, r1_inner=0, nFace=24, frame=transform1)
    vertices2, edges2 = buildConic(height=1, r0_outer=0.5, r0_inner=0, r1_outer=0, r1_inner=0, nFace=24, frame=transform2)

    # TODO:
    # integrate vertices and edges into primitives so that they can be calculated at definition (Primitive.vertices, Primitive.edges)
    # we then have an iterable of all primitives in a module and can draw the modules by iterating through them

    pg.init()
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(60, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, -2.0, -5.0)
    glRotate(-45, 1, 0, 0)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        #glRotate(1, 0.1, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        draw(vertFrame, edgeFrame)
        draw(vertices0, edges0)
        draw(vertices1, edges1)
        draw(vertices2, edges2)
        pg.display.flip() # updates the display
        pg.time.wait(10)



main()