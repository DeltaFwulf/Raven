The planet is a class, representing some celestial body, used to calculate the relative forces acting on a vehicle in flight. It can also perform useful reference frame transformations for placing rockets in the correct location, and calculating expected local conditions given position.
# Derivation of Functions

## Obtaining a Vector with Desired Pitch and Heading

During certain portions of flight, the rocket will be required to point to a specific pitch and heading at its current position. To calculate this vector, the local position and planet shape definition are required.
For a spherical planet, only the local position is required, defined as the position within the planet's PCR or PCNR reference frame. When the planet is non-spherical, either the shape definition to get tangent or the gravitational acceleration vector may be used.

Whatever vector is used, they are used to create a unit vector normal the surface that intersects with the local position vector, and will here be named $\hat{n}$. This is used to create the local East vector tangential to the surface.
$$ \hat{e} = \frac{\hat{n} \times \hat{z}}{\left|\hat{n} \times \hat{z} \right|} $$
Next, heading is set by rotating $\hat{e}$ about $\hat{n}$ by the required angle $\theta =\mod\left(\frac{\pi}{2} - \alpha, 2\pi \right)$, where $\alpha$ is the desired heading (clockwise from North). To get the pitch, a new axis of rotation is required, found as the cross product of the current pointing vector and the normal vector:
$$ \vec{a} = \hat{p} \times \hat{n} $$
The final pointing vector is obtained by rotating $\hat{p}$ by the desired pitch about $a$. It is possible to change the order of these operations by first applying pitch. To do so, the axis of rotation is instead the local south vector: 
$$ \vec{a}_{south} = \hat{e} \times \hat{n} $$
Pitch is then set by rotation of $\hat{e}$ about $\vec{a}_{south}$, then rotated by $\theta$ about $\hat{n}$.

To save computation, it may be helpful to not bother normalising the basis vectors and only normalising the result. This should be fine, as long as the function used for rotation ensures the quaternion is a unit quaternion.


## Obtain Relative Velocity Between Rotating Frame Point and Object in Frame

If the local airspeed of a vehicle is desired, or some object is to be placed with some local velocity on a planet, the inertial velocity of that point within the rotating PCR frame is required. Recall from 3D kinematics that the velocity of a point rotating about an axis is given by:
$$ \vec{v}_{rot} = \vec{\omega} \times \vec{r}$$
The velocity of the object is then subtracted from this velocity for the relative velocity. It is important that the frame is kept consistent between the two velocities. For Raven, the PCNR frame is used, meaning that the velocity of the planet must be subtracted from the velocity of the object.
$$ \vec{v}_{rel} = \vec{v}_{rot} - \vec{v}_{obj} $$
It is important to remember that the result is the velocity of the air from the perspective of the vehicle. This was chosen as the majority of times this function is called will be for drag calculations; this gives the velocity vector in the same direction as the drag force. If this is not wanted, then the relative velocity will instead be changed to $\vec{v}_{rel} = \vec{v}_{obj} - \vec{v}_{rot}$ .
