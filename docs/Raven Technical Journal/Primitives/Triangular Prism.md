A triangular prism is a polyhedron with parallel triangular faces, offset by some thickness with no rotation. The triangular face is defined by three points, defined by their x, y components. One triangular face of the prism lies on the root YZ plane, and the other has a negative x component. The thickness is defined by the distance between the two parallel faces.

In the following derivations, the following symbols are used:
$\rho$ is density (kg / m^3)
$t$ is thickness (m)
$m$ is mass (kg)

The subscript i represents the ith point in the triangle point array (from 1 to 3).
## Constraints
- the thickness must be greater than 0
- the density must be greater than 0
- the triangular face must have an area greater than 0
- density is constant throughout the prism

# Derivation of Inertial Properties

## Point Order and Rotation
- integral is for left point case
- how to make any triangle left point
- reverse point order if parity case


## Mass
The mass of a triangular prism is calculated by $m = \rho t A$. The area is calculated as:
$$ A = \frac{1}{2}\left|(y_1 - y_3)(z_2 - z_1) - (y_1 - y_2)(z_3 - z_1)  \right|$$
If the points are arranged in counter clockwise order about some point inside the triangle, the absolute value operation is not necessary.

## Centre of Mass

The centre of mass lies halfway along the thickness, and at the centre of area of the triangle. The centre of area is the average location of the three points:
$$ (\bar{y}, \bar{z}) = \sum^3_{(i=1)} \frac{1}{3}p_i $$

The centre of mass is expressed as a vector relative to the root frame:
$$ c = \left(-\frac{t}{2}, \bar{y}, \bar{z}\right)$$
## Moment of Inertia

The following derivation applies to triangles that contain two points with shared y component, and the third point having a lower y component than the shared value, i.e. 'left-pointing'. This is achieved by passing the triangle after rotation and point re-ordering as explained in a previous section.

To find the moment of inertia about a principal axis, once solves a triple integral for dI about this axis:

$$ I_{xx} = \int_{x_f}^{x_0}\int_{y_0}^{y_f}\int_{z_0(y)}^{z_f(y)}y^2 + z^2 dz.dy.dx$$
$$ I_{yy} = \int_{x_f}^{x_0}\int_{y_0}^{y_f}\int_{z_0(y)}^{z_f(y)}x^2 + z^2 dz.dy.dx$$
$$ I_{zz} = \int_{x_f}^{x_0}\int_{y_0}^{y_f}\int_{z_0(y)}^{z_f(y)}x^2 + y^2 dz.dy.dx$$

The other terms in the tensor, as solution is at principal axes, are all 0. The total tensor may be expressed in this form:

$$I = \rho\begin{bmatrix}F + G & 0 & 0\\0 & G + H & 0\\0 & 0 & F + H\end{bmatrix}$$

where the integrals F, G, and H represent x, y, and z inertia contributions, respectively:

$$ F = t\left[\frac{k_f - k_0}{3} \left(y_f^3 - y_0^3\right) + \frac{s_f - s_0}{4}\left(y_f^4 - y_0^4\right)\right]$$
$$G = t\left[\left(k_f^3 - k_0^3\right)\left(y_f - y_0\right) + \frac{3}{2}\left(k_f^2s_f - k_0^2s_0\right)\left(y_f^2 - y_0^2\right) + \left(k_fs_f^2 - k_0s_0^2\right)\left(y_f^3 - y_0^3\right) + \frac{s_f^3 - s_0^3}{4}\left(y_f^4 - y_0^4\right)\right] $$
$$ H = \frac{t^3}{12}\left[\left(k_f - k_0\right)\left(y_f - y_0\right)  + \frac{s_f - s_0}{2}\left(y_f^2 - y_0^2\right)\right] $$

The constants of integration, $k_f$, $k_0$, $s_f$, and $s_0$, are defined as follows:

$$ s_f = \frac{z_1 - z_2}{y_f - y_0}$$
$$ s_0 = \frac{z_3 - z_2}{y_f - y_0} $$
$$ k_f = z_2 - s_fy_0 $$
$$ k_0 = z_2 - s_0y_0 $$

And the y limits of integration, $y_0$ and $y_f$ and the minimum and maximum y components in the triangle, by definition: $y_0 = y_2$ and $y_f = y_1 = y_3$. Combinations of constants of integration can return them back in terms of these limits:

$$ s_f - s_0 = \frac{z_1 - z_3}{y_f - y_0} $$
$$ k_f - k_0 = \left(z_2 - s_fy_0\right) - \left(z_2 - s_0y_0\right)$$
$$ k_f - k_0 = y_0\left(s_0 - s_f\right) $$
$$ k_f - k_0 = y_0 \frac{z_3 - z_1}{y_f - y_0} $$
This can be used to simplify several terms, for example:
$$ \left(k_f - k_0\right)\left(y_f - y_0\right) = y_0\left(z_3 - z_1\right) $$
$$ \left(s_f - s_0\right)\left(y_f^2 - y_0^2\right) = \left(z_1 - z_3\right)\left(y_f + y_0\right)$$

Here are the three main interial terms, with constants of integration removed:
$$ F = \frac{t}{12}\left(z_1 - z_3\right)\left(3y_f^3 - y_0y_f^2 - y_0^2y_f - y_0^3\right) $$
$$ H = \frac{t^3}{24}\left(z_1 - z_3\right)\left(y_f - y_0\right) $$
