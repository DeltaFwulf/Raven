import numpy as np



def mesh2d(pts:list[np.ndarray]) -> list[int]:
    """Calculates a triangular mesh for a polygon defined by adjacent points, in ascending order for CCW representation."""
  
    shapes = []
    newShapes = [[i for i in range(len(pts))],]

    # using a global points set, shapes are represented by their set of indices within the point set
    while len(newShapes) != len(shapes):

        shapes = newShapes
        newShapes = []

        for shape in shapes:
            newShapes += splitShape(pts=pts, c=shape)
 
    # triangulate each monotonic polygon and combine meshes
    tris = []
    for m in shapes:
        tris += triangulate(m)

    return tuple(tris)



def splitShape(pts_global:list[np.ndarray], c:list[int]) -> list[list[int]]:
    """Given some polygon, splits the shape into a monotonic subshape and remaining shapes, returning the new set of shapes.
       If the shape is already monotonic, returns only the input shape."""
    
    # c is the set of points, counterclockwise from the start point.
    N = len(c)
    pts = [pts_global[i] for i in c]
    
    cy = np.flip(np.argsort([float(pt[1]) for pt in pts])) # the order in which to access c array to give descending y values of points in sub shape
    edges = [[c[i], c[(i + 1) % N]] for i in range(N)]
    
    x0 = pts[0][0] # ensures the bounds are contained within the shape's true bounds (not strictly necessary tbh) (learned from drawFrames)
    xf = pts[0][0]
    for pt in pts: # sweep line width (x)
        x0 = min(pt[0], x0)
        xf = max(pt[0], xf)

    subshapes = []
    
    for i in range(N):

        ky = cy[i]
        y = pts[ky][1]

        edges_low = []
        for edge in edges:
            if pts[edge[0]][1] < y or pts[edge[1]][1] < y:
                edges_low.append(edge)

        pts_sweep = pts_global + [np.array([x0, y], float), np.array([xf, y], float)]
        intersectingEdges = intersects(pts=pts_sweep, set_a=[[-2, -1],], set_b=edges_low)

        if len(intersectingEdges) > 2: # shape is not monotonic, split along all previous points and current point
            
            l0 = c[ky]
            r0 = c[ky]

            lf = l0 - 2
            rf = r0 + 2

            # one direction must join a point with higher index (c) and the other lower index (c) than ky
            # target is the next value above c[ky] that is lower in cy
            leftConnected = False
            rightConnected = False

            while not leftConnected:
                for j in range(l0 - 2, -1, -1):
                    if cy[j] < ky:
                        lf = j

                if len(intersects(pts=pts_global, set_a=[[l0, lf],], set_b=edges)) == 4:
                    leftConnected = True
                else:
                    l0 -= 1

            while not rightConnected:
                for j in range(l0 + 2, N):
                    if cy[j] < ky:
                        rf = j

                if len(intersects(pts=pts_global, set_a=[[r0, rf],], set_b=edges)) == 4:
                    rightConnected = True
                else:
                    r0 += 1

            # uppper shape ########################################################################################################
            subshapes.append([lf,] + [u for u in range(l0, r0 + 1)] + [u % N for u in range(rf, N + lf)])
            subshapes.append([u for u in range(lf, l0 + 1)])
            subshapes.append([u for u in range(r0, rf + 1)])
        
            return subshapes # only make the split once per function run

    if subshapes == []:
        subshapes.append(c)

    return subshapes

        

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



def triangulate(pts:list[np.ndarray]) -> list[list[int]]:
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

    return tris