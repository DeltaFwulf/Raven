import numpy as np



def mesh2d(pts:list['np.ndarray']) -> list['int']:
    """Calculates a triangular mesh for a polygon defined by adjacent points"""

    monotonics = []
    tris = []

    # define sweep bounds
    xMin = 0
    xMax = 0
    yMin = 0
    yMax = 0

    for pt in pts:
        xMin = min(pt[0], xMin)
        xMax = max(pt[0], xMax)
        yMin = min(pt[1], yMin)
        yMax = max(pt[1], yMax)

    sweeper = [np.array([xMin, yMax], float), np.array([xMax, yMax], float)]

    # for each point's y, check for intersections between all polygon lines
    

    # if this is detected, the shape must be split to make monotonic
    # at which nodes should the shape be split?

    # start sweep for truncated shape from new top

    # triangulate each monotonic polygon
    tris = []
    for m in monotonics:
        tris += triangulate(m)

    return tris



# TODO: solve for vertical lines (include a unit test)
def intersects(pts:list[np.ndarray], ia:list[int], ib:list[int]) -> bool:
    """Given a set of points and two sets of connections (expressed by index), 
       calculates whether at least one intersection exists between the two sets of lines."""
    
    for a in range(len(ia)):
        for b in range(len(ib)):

            la = a[0] if pts[a[0]][0] < pts[a[1]][0] else a[1]
            ra = a[1] if a == a[0] else a[0]
            lb = b[0] if pts[b[0]][0] < pts[b[1]][0] else b[1]
            rb = b[1] if b == b[0] else b[0]

            if ra < lb or la > rb: # overlap cannot occur
                continue

            il = max(la, lb)
            ir = min(ra, rb)

            xl = pts[il][0]
            xr = pts[ir][0]
      
            if pts[la][0] == pts[ra][0]:
                pla = pts[la]
                pra = pts[ra]

            else:
                pla = pts[la] if il == la else pts[la] + (pts[ra] - pts[la])*(pts[la][0] - xl) / (xr - xl)
                pra = pts[ra] if ir == ra else pts[ra] + (pts[ra] - pts[la])*(pts[ra][0] - xr) / (xr - xl)
            
            if pts[lb][0] == pts[rb][0]:
                plb = pts[lb]
                prb = pts[rb]
            
            else:
                plb = pts[lb] if il == lb else pts[lb] + (pts[rb] - pts[lb])*(pts[lb][0] - xl) / (xr - xl)
                prb = pts[ra] if ir == rb else pts[rb] + (pts[rb] - pts[lb])*(pts[rb][0] - xr) / (xr - xl)

            dl = pla - plb
            dr = pra - prb
            
            if np.dot(dl, dr) <= 0:
                return True
            
    return False



# TODO: determine if ccw always works, or if descending monotonic height required
def triangulate(pts:list[np.ndarray]) -> tuple[int]:
    """Given a monotonic polygon, split into triangles."""

    N = len(pts)
    edges = [[i, (i + 1) % N] for i in range(N)] # stores edges of polygon perimeter
    branches = [] # stores non-adjacent connections
    dead = [] # corners are 'dead' if they can no longer form connections
    tris = []

    for i in range(N):

        cons = [edges[i], edges[i - 1]]
        
        for j in range(1, N - 1):

            if j in dead:
                continue

            branch = [i, (i + j) % N]

            if not intersects(pts=pts, ia=branch, ib=edges + branches):
                branches.insert(-2, branch)
                cons.append(branch)

            for k in range(i, (i + N) % N): # TODO: decide if this is worth doing, and cross over point when to skip
                
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
                
                if inTri and not inBranch:
                    dead.append(k)

        for j in range(len(cons) - 1): # form triangles from consecutive branches
            b3 = [cons[j][1], cons[j + 1][1]]
            if b3 not in branches and [b3[1], b3[0]] not in branches:
                branches.append(b3)
           
            tris.append((i, b3[0], b3[1]))

    return tris