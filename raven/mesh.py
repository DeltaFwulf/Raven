import numpy as np
from collections import OrderedDict


def mesh2d(pts:list[np.ndarray]) -> list[int]:
    """Calculates a triangular mesh for a polygon defined by adjacent points, in ascending order for CCW representation."""
  
    shapes = []
    newShapes = [[i for i in range(len(pts))],]

    # using a global points set, shapes are represented by their set of indices within the point set
    while len(newShapes) != len(shapes):

        shapes = newShapes
        newShapes = []

        for shape in shapes:
            newShapes += splitShape(pts=pts, ind=shape)
 
    # triangulate each monotonic polygon and combine meshes
    tris = []
    for m in shapes:
        tris += triangulate(m)

    return tris



def splitShape(pts_global:list[np.ndarray], ind:list[int]) -> list[list[int]]:
    """Given some polygon, splits the shape into a monotonic subshape and remaining shapes, returning the new set of shapes.
       If the shape is already monotonic, returns only the input shape."""
    
    # FIXME: only check edges that are at or below the point in question
    # FIXME: do not count non-monotonic if any line is horizontal (this will fail currently)


    # Create the subset of points and edges within this shape
    N = len(ind)
    edges = [[ind[i], ind[(i + 1) % N]] for i in range(N)]
    corners = [pts_global[i] for i in ind]
 
    x0 = 0
    xf = 0
    for corner in corners: # sweep line width (x)
        x0 = min(corner[0], x0)
        xf = max(corner[0], xf)

    cy = np.argsort([cn[1] for cn in corners])
    subs = []
    
    for i in range(N):

        c = cy[i]
        pt_set = corners + [np.array([x0, corners[c][1]], float), np.array([xf, corners[c][1]], float)]
        intersectingEdges = intersects(pts=pt_set, set_a = [[-2, -1],], set_b=edges)

        if len(intersectingEdges) > 2: # shape is not monotonic, split along all previous points and current point
            
            pt_left = cy[i - 1]
            pt_right = intersectingEdges[-1][1]
            
            if len(intersects(pts=corners, set_a=[[c, pt_right],], set_b=edges)) > 0:
                pt_right = intersectingEdges[-1][0]
                # FIXME: catch double block case

            subs.append([u for u in range(pt_left)] + [u for u in range(pt_left, pt_right)] + [u for u in range(pt_right, -1)]) # upper shape
            subs.append([u for u in range(pt_left, c)]) # left side bottom shape
            subs.append([u for u in range(pt_right, -1)]) # right side bottom shape

        else:
            subs.append(edges)

    return subs

    

def intersects(pts:list[np.ndarray], set_a:list[int], set_b:list[int]) -> list[list[int]]:
    """Given a set of points and two sets of edges (expressed by point index),
       return the subset of edges in b that intersect with at least one edge in a."""
    
    intersections = []
    
    for a in set_a:
        for b in set_b:

            la = a[0] if pts[a[0]][0] < pts[a[1]][0] else a[1]
            ra = a[1] if la == a[0] else a[0]
            lb = b[0] if pts[b[0]][0] < pts[b[1]][0] else b[1]
            rb = b[1] if lb == b[0] else b[0]

            if pts[ra][0] < pts[lb][0] or pts[la][0] > pts[rb][0]: # lines cannot cross
                continue

            # define overlap boundary
            xla = pts[la][0]
            xlb = pts[lb][0]
            xl = max(xla, xlb)

            xra = pts[ra][0]
            xrb = pts[rb][0]
            xr = min(xra, xrb)

            if pts[la][0] == pts[ra][0]:
                pla = pts[la]
                pra = pts[ra]

            else:
                pla = pts[la] if xl == xla else pts[la] + (pts[ra] - pts[la])*(xl - xla) / (xra - xla)
                pra = pts[ra] if xr == xra else pts[ra] + (pts[la] - pts[ra])*(xr - xra) / (xla - xra)
    
            if pts[lb][0] == pts[rb][0]:
                plb = pts[lb]
                prb = pts[rb]
            
            else:
                plb = pts[lb] if xl == xlb else pts[lb] + (pts[rb] - pts[lb])*(xl - xlb) / (xrb - xlb)
                prb = pts[rb] if xr == xrb else pts[rb] + (pts[lb] - pts[rb])*(xr - xrb) / (xlb - xrb)

            dl = pla - plb
            dr = pra - prb
            
            if np.dot(dl, dr) <= 0:
                intersections.append(b)
            
    return intersections



def triangulate(pts:list[np.ndarray]) -> tuple[list[int]]:
    """Given a monotonic polygon, split into triangles."""

    N = len(pts)
    edges = [[i, (i + 1) % N] for i in range(N)] # stores edges of polygon perimeter
    
    if N == 3: # the polygon is already a triangle
        return [edges[i][0] for i in range(N)]
    
    branches = [] # stores non-adjacent connections
    dead = [] # corners are 'dead' if they can no longer form connections
    tris = []

    for i in range(N):
        cons = [edges[i], edges[i - 1]]
        
        for j in [a % N for a in range(i + 2, i + N - 1)]:

            if j in dead:
                continue

            branch = [i, j]

            if len(intersects(pts=pts, set_a=[branch,], set_b=edges + branches)) == 4:
                branches.append(branch)
                cons.insert(-1, branch)
                
            for k in [(i + n) % N for n in range(N)]:
                inTri = False
                inBranch = False

                for tri in tris:
                    if k in tri:
                        inTri = True
                        break

                for branch in branches:
                    if k in branch:
                        inBranch = True
                        break
                
                if inTri and not inBranch and k not in dead:
                    dead.append(k)

        if len(cons) < 3:
            continue

        for j in range(len(cons) - 1): # form triangles from consecutive branches

            adj = cons[j] + cons[j + 1] # adjacent connections
            b3 = [p for p in adj if adj.count(p) == 1]

            if (b3 not in branches) and (b3 not in edges) and ([b3[1], b3[0]] not in branches) and ([b3[1], b3[0]] not in edges):
                branches.append(b3)

            newTri = (i, b3[0], b3[1])
            triIsNew = True
            for tri in tris:
                if all(k in tri for k in newTri):
                    triIsNew = False
                    break

            if triIsNew:
                tris.append(newTri)

    return tuple(tris)