This shape is the generalised form of the conic shape (we remove the assumption that the shape is solid, as in the [[Conic Full]] shape. The constraint placed on the internal profile of the object is that it must be definable by a line segment with a range of {0, R(x)}, with its ends at {x0, xf}. This prevents the shape from having negative volume. 

## Mass
The mass of the general conic shape is given by a boolean operation of two conic full shapes as:

$$
	\frac{\pi \rho l}{3}\left(\hat{R}_f^2 + \hat{R}_0R_f + \hat{R}_0^2\right) 
	- \frac{\pi\rho l}{3}\left(\check{R}_f^2 + \check{R}_0\check{R}_f + \check{R}_0^2\right)
$$
which factorises to

$$
\frac{\pi\rho l}{3}\left((\hat{R}_f^2 - \check{R}_f^2) + (\hat{R}_0\hat{R}_f - \check{R}_0\check{R}_f) + (\hat{R}_0^2 - \check{R}_0^2)\right)
$$


## Centre of Mass

The centre of mass of the object can be found similarly by sweeping an annulus from x = 0 to x = l:

$$
\bar x = \frac{x_f}{4}\left(\frac

{(\hat R_0^2 - \check R_0^2) + 2(\hat R_0 \hat R_f - \check R_0 \check R_f) + 3(\hat R_f^2 - \check R_f^2)}

{(\hat{R}_f^2 - \check{R}_f^2) + (\hat{R}_0\hat{R}_f - \check{R}_0\check{R}_f) + (\hat{R}_0^2 - \check{R}_0^2)}

\right)
$$

## Inertia Tensor

The inertia tensor can be calculated from two locations.

**a)** calculating I from the centre of mass
**b)** calculating I from the root of the shape

method **b** would be simpler algebraically, and should allow the calculation of I about the centre of mass by invoking parallel axis theorem. We can derive the conic full I tensor from the node and then perform a boolean operation to get the final tensor.

However, from inspection of the inertia tensor of the conic full section, the derivation is equivalent to subtracting the smaller full conic from the larger conic, which is valid so long as the shapes have the same limits in x (we can choose these limits arbitrarily along the x axis). 

 $$I_{yy} = \rho(\hat A - \check A + \hat B - \check B)$$

We can take the origin of the part to be the centre of mass, as this means that we do not have to determine if the centre of rotation is closer to the centre of mass than the primitive origin. This allows us to use the same function to calculate the transformed tensor.




