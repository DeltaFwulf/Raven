To solve the inertial properties of the fin, it has been decdied to split the polygonal fin profile into triangles before extruding them into prisms and solving the mindividually.
The solution to the meshing problem has three main components:
- Detecting intersections between sets of lines
- Splitting general polygons into monotonic polygons
- Triangulating monotonic polygons

Once the monotonic subshapes have been meshed, the total mesh is reconstructed by combining all submeshes, removing duplicate points from the total list.

# Detecting Line Intersections
There are multiple ways to detect intersections between lines. The first is a naive approach, which is likely sufficiently efficient for this application (the polygons will not have many triangles). The other is an optimal algorithm, which may be implemented if development times permit.

# Splitting polygons into monotonic subshapes
A monotonic polygon is one which does not 'reverse' direction as the path is followed from top to bottom point over one axis. To determine if a polygon is monotonic, a line sweep method is employed, whereby a line perpendicular to the axis of interest is swept along said axis from one extremum to the other. If the polygon is monotonic, this line will intersect at most two lines on the boundary; otherwise, a reversal exists and the shape must be split. To split this line, the shape is split at the ends of the offending lines until subshapes are monotonic.


# Triangulation of Monotonic Polygons
Two methods have been devised to triangulate monotonic polygons: *corner counting* and *ascending connections*. As the former is a less efficient method than the latter, fan sweep will be explored first.

## Ascending Connections Method
To triangulate the polygon, lines must be drawn between non adjacent corners until the entire area is covered by non intersecting triangles. This can be done by starting at the first (or any) point, then attempting to form line connections between all non-adjacent corners.  If a line intersects with any existing line, it is rejected. This process is repeated for all corner points until no further lines may be formed.

To obtain the triangle corner indices, the sequence of connections made is used. For example, if one begins with a square with corners c1, c2, c3, and c4, where c1 is adjacent to c2 and c4, the only new line it can form is c1-c3. Now, the lines connected to c1 are checked in ascending order.

c1 -> c2, c1-> c3, c1-> c4

To form the first triangle, the lines c1 -> c2 and c1 -> c3 form the first two sides. The function then checks if a line exists between c2 and c3, creating one if not. This step should never create a line which intersects another, as if another point lay inside the triangle that would form, this point would be the second side of the triangle.

This forms the first triangle (c1, c2, c3).

Next, the lines c1 -> c3 and c1 -> c4 are used. The process is repeated, and the third line is given as c3 -> c4, to form the second triangle (c1, c3, c4).

Since no other corners are the origin of a connecting line, then the mesh is completed. This would be redundant, as duplicate triangles would form for all other points.


### Corner Bracketing

It is possible to determine when a point on the polygon's perimeter can no longer form any connections that do not intersect existing lines. This method has been called bracketing, and will be used to skip line connection attempts when meshing

*Description of algorithm here*


