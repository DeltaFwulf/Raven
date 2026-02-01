A function that can determine intersections between sets of line segments is needed to create 2D triangular meshes within Raven (link here).

This function must take in a set of points, as well as two arrays of edges, where an edge is defined by two indices. The edge is the line segment between the two indices. This function must work for vertical line segments, as well as non-vertical. The two sets of edges may contain shared elements, and duplicates may exist in each set (duplicate intersections are counted as multiple intersections).

- If an identical line exists in set A and set B, they are said to intersect.
- If lines contain a shared point, they intersect.

The function should output a list of all line segments in set B that intersect with any line segments in set A.

# Determining if Two Line Segments Intersect

Let two line segments be defined as a and b, both represented by their end point vectors. The left and right points of the segments are found from their x components. If the segments are vertical, the ordering is arbitrary. This gives two points for each segment: $l_i$ and $r_i$, where i = a or b for corresponding segment.

Firstly, a check is made to see if a and b overlap in x. If not, they cannot intersect. Therefore, no intersection can exist for the case where $l_a > r_b | r_a < l_b$.

Otherwise the overlap x boundaries are calculated:
$x_l = max(l_a, l_b)$
$x_r = min(r_a, r_b)$

And the y component of the line segments calculated at $x_l$ and $x_r$:

$$y_{i}(x_l) = (y_{r,i} - y_{l,i})*\frac{x_{r,i} - x_l}{x_{r_i} - x_{l,i}}$$
$$y_{i}(x_r) = (y_{l,i} - y_{r,i})*\frac{x_{l,i} - x_r}{x_{l_i} - x_{r,i}}$$
And the difference vectors calculated at $x_l$ and $x_r$:

$$\Delta_l = (0, y_a(x_l) - y_b(x_l))$$
$$\Delta_r = (0, y_a(x_r) - y_b(x_r))$$

If the lines do not intersect, then over the overlap boundary, one line must stay either above or below the other without crossing. Therefore, the dot product between the two difference vectors must be > 0, as they are parallel vectors. If the lines touch at exactly the boundary, one difference vector will be 0. Otherwise, the vectors will be facing opposite directions, giving a negative dot product. Therefore, lines intersect if $\Delta_a \cdot \Delta_b \le 0$.


