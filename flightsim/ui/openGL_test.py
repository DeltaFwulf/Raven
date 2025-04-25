"""Let's attempt to draw a wireframe shape, then build this file into a part visualiser

The objective of this code is to be able to take in a shape definition (module, primitives, etc) and a location,
and draw the shape in 3D in a window.

"""

import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import gluPerspective
import numpy as np
from math import cos, sin, pi

# create a list of vertices for a conic() shape:
# currently, it is a 1x1 right cylinder:
# for future reference, this will be run at initialisation of each primitive, then stored in each object as tuples
def getConicVertices(height:float, r0_outer:float, r0_inner:float, r1_outer:float, r1_inner:float, curvFaces:int):
    """
    Vertex order: outer ring 0, outer ring 1, inner ring 0, inner ring 1
    
    
    """

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

    offset_inner0 = 2 * curvFaces - 1
    offset_inner1 = 3 * curvFaces if r0_inner > 0 else 2 * curvFaces

    vertices = []
    edges = []

    ang = np.linspace(0, 2 * pi, curvFaces)

    # create outer conic vertices
    for i in range(0, curvFaces):

        v0_outer.append((r0_outer * cos(ang[i]), r0_outer * sin(ang[i]), x0))
        v1_outer.append((r1_outer * cos(ang[i]), r1_outer * sin(ang[i]), xf))

        edges.append((i, i+1)) # ring 0
        edges.append((i + curvFaces, i + curvFaces + 1)) # ring 1
        edges.append((i, i + curvFaces)) # ring 0 to ring 1

    vertices.extend(v0_outer)
    vertices.extend(v1_outer)

    # we have 4 configurations of the inner surface:
    # 1): there is no inner hole (inners both equal 0) (no inner vertices)
    # 2): lower is 0, upper is non-zero (many to point)
    # 3): upper is 0, lower is non-zero (many to point)
    # 4): both are non-zero (many to many)

    if(r0_inner > 0 and r1_inner > 0):
        for i in range(0, curvFaces):
            
            v0_inner.append((r0_inner * cos(ang[i]), r0_inner * sin(ang[i]), x0))
            v1_inner.append((r1_inner * cos(ang[i]), r1_inner * sin(ang[i]), xf))

            edges.append((i + offset_inner0, i + offset_inner0 + 1)) # inner ring 0
            edges.append((i + offset_inner1, i + offset_inner1 + 1)) # inner ring 1
            edges.append((i + offset_inner0, i + offset_inner1)) # inner ring 0 -> inner ring 1

        vertices.extend(v0_inner)
        vertices.extend(v1_inner)
            

    elif(r0_inner == 0 and r1_inner > 0):
        
        v0_inner.append([0, 0, x0])

        for i in range(0, curvFaces):
            v1_inner.append((r1_inner * cos(ang[i]), r1_inner * sin(ang[i]), xf))
            edges.append((offset_inner0, i + offset_inner1))

        vertices.extend(v0_inner)
        vertices.extend(v1_inner)

    elif(r0_inner > 0 and r1_inner == 0):

        v1_inner.append([0, 0, xf])

        for i in range(0, curvFaces):
            v0_inner.append((r0_inner * cos(ang[i]), r0_inner * sin(ang[i]), x0))
            edges.append((i + offset_inner0, i + offset_inner0 + 1))
            edges.append((i + offset_inner0, offset_inner1))

        vertices.extend(v0_inner)
        vertices.extend(v1_inner)

    #vertices = tuple(vertices) # XXX: this is probably not ideal if we have to transform relative to other parts later on
    #edges = tuple(edges)

    return vertices, edges


"""
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
"""




def shape(vertices:tuple, edges:tuple):

    glBegin(GL_LINES)
    
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])

    glEnd()


def main():

    vertices, edges = getConicVertices(height=1, r0_outer=0.5, r0_inner=0, r1_outer=0.25, r1_inner=0.1, curvFaces=8)

    pg.init()
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)

    #gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5.0)

    while True:

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        glRotate(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        shape(vertices, edges)
        pg.display.flip() # updates the display
        pg.time.wait(10)



main()