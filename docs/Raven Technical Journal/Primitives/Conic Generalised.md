This shape is the generalised form of the conic shape (we remove the assumption that the shape is solid, as in the [[Conic Full]] shape. The constraint placed on the internal profile of the object is that it must be definable by a line segment with a range of {$0, \check R(x)$} with its ends at {$x_0, x_f$} This prevents the shape from having negative volume. 

# Mass
The mass of the general conic shape is given by a boolean operation of two conic full shapes as:

$$
	\frac{\pi \rho l}{3}\left(\hat{R}_f^2 + \hat{R}_0R_f + \hat{R}_0^2\right) 
	- \frac{\pi\rho l}{3}\left(\check{R}_f^2 + \check{R}_0\check{R}_f + \check{R}_0^2\right)
$$
which factorises to
$$
\frac{\pi\rho l}{3}\left((\hat{R}_f^2 - \check{R}_f^2) + (\hat{R}_0\hat{R}_f - \check{R}_0\check{R}_f) + (\hat{R}_0^2 - \check{R}_0^2)\right)
$$
# Centre of Mass

The centre of mass of the object can be found similarly by sweeping an annulus from x = 0 to x = l:

$$
\bar x = \frac{x_f}{4}\left(\frac

{(\hat R_0^2 - \check R_0^2) + 2(\hat R_0 \hat R_f - \check R_0 \check R_f) + 3(\hat R_f^2 - \check R_f^2)}

{(\hat{R}_f^2 - \check{R}_f^2) + (\hat{R}_0\hat{R}_f - \check{R}_0\check{R}_f) + (\hat{R}_0^2 - \check{R}_0^2)}

\right)
$$
# Moment of Inertia

The inertia tensor can be calculated from two locations.

**a)** calculating I from the centre of mass
**b)** calculating I from the root of the shape

method **b** would be simpler algebraically, and should allow the calculation of I about the centre of mass by invoking parallel axis theorem. We can derive the conic full I tensor from the node and then perform a boolean operation to get the final tensor.

We can take the origin of the part to be the centre of mass, as this means that we do not have to determine if the centre of rotation is closer to the centre of mass than the primitive origin. This allows us to use the same function to calculate the transformed tensor.

### $I_{xx}$
$$
I_{xx} = \rho \int_{x_0}^{x_f} \int_{0}^{2\pi} \int_{\check R(x)}^{\hat R(x)} r^3 \ dr d\theta dx
$$

By integrating with respect to r, we see that the moment of inertia is equivalent to the difference of two full cones (equation of an annulus as well but this will come in handy later)

$$
I_{xx} = \rho \int_{x_0}^{x_f} \int_{0}^{2\pi} \hat R^4(x) - \check R^4(x) \ d\theta dx
$$
$$
I_{xx} = \rho \int_{x_0}^{x_f} \int_{0}^{2\pi} \hat R^4(x) \ d\theta dx - \rho \int_{x_0}^{x_f} \int_{0}^{2\pi} \check R^4(x) \ d\theta dx
$$

and so:
$$
I_{xx} = \hat I_{xx} - \check I_{xx}
$$


Where each integral takes the form:
$$
Ixx = \frac{\pi\rho (x_f- x_0)}{10}\frac{R_f^5 - R_0^5}{R_f - R_0}\quad\rvert\quad R_f\neq R_0
$$ $$
	I_{xx} = \frac{\pi\rho x_f}{2}R^4 \quad \rvert \quad R_0 = R_f = R
$$
### $I_{yy}, I_{zz}$

Similarly, from inspection of the inertia tensor of the conic full section, the derivation is equivalent to subtracting the smaller full conic from the larger conic.
 $$I_{yy} = \rho(\hat A - \check A + \hat B - \check B)$$
or more simply:

$$
I_{yy} = \hat I_{yy} - \check I_{yy}
$$

Each term takes the form:
$$
A = \frac{\pi}{3} \left(

\frac{3k^2}{5}    \left(x_f^5 - x_0^5\right)
+
\frac{3k}{2}(R_0 - kx_0)    \left(x_f^4 - x_0^4\right)
+
(R_0 - kx_0)^2    \left(x_f^3 - x_0^3\right)

\right)
$$
$$
B = \frac{\pi(x_f-x_0)}{20}\left[

5R_0^4 + 10R_0^3(R_f-R_0) + 10R_0^2(R_f-R_0)^2+5R_0(R_f-R_0)^3 + (R_f-R_0)^4

\right]
$$

$x_0$ and $x_f$ are relative to the combined shape's centre of mass. As this shape is axisymmetric, we can also say that $I_{zz} = I_{yy}$.



# A Method of Two Cones

- using two cones, all properties may be found potentially more simply