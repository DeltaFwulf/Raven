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

# create a list of vertices for a conic() shape:
# currently, it is a 1x1 right cylinder:
# for future reference, this will be run at initialisation of each primitive, then stored in each object as tuples
def getConicVertices(height:float, r0_outer:float, r0_inner:float, r1_outer:float, r1_inner:float, curvFaces:int):

    v0_outer = []
    v0_inner = []

    v1_outer = []
    v1_inner = []

    ang = np.linspace(0, 2 * pi, curvFaces)

    # create outer conic vertices
    for i in range(0, curvFaces):

        v0_outer.append([r0_outer * cos(ang[i]), r0_outer * sin(ang[i]), 0])
        v1_outer.append([r1_outer * cos(ang[i]), r1_outer * sin(ang[i]), height])

    v0_outer = tuple(v0_outer)
    v1_outer = tuple(v1_outer)

    # we have 4 configurations of the inner surface:
    # 1): there is no inner hole (inners both equal 0) (no inner vertices)
    # 2): lower is 0, upper is non-zero (many to point)
    # 3): upper is 0, lower is non-zero (many to point)
    # 4): both are non-zero (many to many)

    if(r0_inner > 0):

        if(r1_inner > 0):
            for i in range(0, curvFaces):
                v0_inner.append([r0_inner * cos(ang[i]), r0_inner * sin(ang[i]), 0])
                v1_inner.append([r1_inner * cos(ang[i]), r1_inner * sin(ang[i]), height])







# cube vertices:
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


def Cube():

    glBegin(GL_LINES)
    
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])

    glEnd()


def main():

    pg.init()
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -5.0)

    while True:

        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        glRotate(1, 3, 1, 1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()
        pg.display.flip() # updates the display
        pg.time.wait(10)


main()