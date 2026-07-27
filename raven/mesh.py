import numpy as np



def mesh2d(pts:list['np.ndarray']) -> list['int']:
    """Calculates a triangular mesh for a polygon defined by adjacent points"""

    # split into monotone polygons

    # triangulate

    # merge meshes

    pass



def checkIntersections():
    pass


def triangulate(pts:list) -> tuple['list']:
    """Given a monotonic polygon, split into triangles."""

    lines = []
    connectable = [i for i in range(len(pts))] # list of corners still available to form connections from some point
    tris = []

    # add all polygon outline lines to the list
    for i in range(len(pts)):
        lines.append([i, (i + 1) % len(pts)])

    for a in range(len(pts)):

        for b in range((a + 2) % len(pts), (a - 2) % len(pts)):

            if b not in connectable:
                continue
            
            con = [pts[a], pts[b]]

            # does this line intersect
            if not checkIntersections(lines, con):
                lines.append(con)

            # create the next triangle (two lines in the shape where a is the origin):



            
            # bracket off a corner if required
                # point is part of a triangle and adjacent to point a

