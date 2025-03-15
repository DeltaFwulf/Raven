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
from flightsim.motion.vectorUtil import referenceFrame
from rocket.primitives import Conic, RectangularPrism
from rocket import materials



def axes(transform:referenceFrame):

    vertices = (transform.local2parent((0, 0, 0)),
                transform.local2parent((1, 0, 0)),
                transform.local2parent((0, 1, 0)),
                transform.local2parent((0, 0, 1)))

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

    homeTransform = referenceFrame(np.array([0, 0, 0], float), angInit=0, axisInit=np.array([1,0,0], float))
    conicTransform = referenceFrame(np.array([2, 0, 0], float), angInit=pi/4, axisInit=np.array([1,1,1], float))
    rectTransform = referenceFrame(np.array([1, 0, 0], float), angInit=0, axisInit=np.array([1,0,0], float))

    # build the shapes from primitives, then draw them according to their positions:
    conic = Conic(length=1, moduleTransform=conicTransform, dOuterRoot=1, dOuterEnd=1, dInnerRoot=0.5, dInnerEnd=0.5, name='conic0', material=materials.Aluminium)
    rectPrism = RectangularPrism(x=1, y=1, z=1, transform=rectTransform, name='rectangular-prism-0', material=materials.Aluminium)
    
    homeVerts, homeEdges = axes(homeTransform)
    conicAxVerts, conicAxEdges = axes(conicTransform)
    rectAxVerts, rectAxEdges = axes(rectTransform)

    pg.init()
    display = (800, 600)
    pg.display.set_mode(display, DOUBLEBUF|OPENGL)

    # Set initial camera position:
    gluPerspective(60, (display[0]/display[1]), 0.1, 50.0)
    glTranslatef(0, -1, -6)
    glRotate(90, 0, 0, 1)

    while True:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                quit()

        glRotate(0.5, 1, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        draw(conic.vertices, conic.edges)
        draw(rectPrism.vertices, rectPrism.edges)
        draw(homeVerts, homeEdges)
        draw(conicAxVerts, conicAxEdges)
        draw(rectAxVerts, rectAxEdges)

        pg.display.flip() # updates the display
        pg.time.wait(10)



main()