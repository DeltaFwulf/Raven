
We will specify rotations using the unit quaternion (this is convenient as we will pre-compute all torques acting on the vehicle relative to its body coordinates and then so find the angular motion of the system
about a single axis passing through the rocket's current centre of mass). 

This rotation will be intrinsic as we are rotating a second coordinate system, so to find the new frame orientation relative to the fixed world axes we must:

Since we want to rotate in vehicle axes, we can treat this as a two step intrinsic rotation:
1) transform the rocket axes relative (in the rocket's body frame)
2) transform the rocket axes to world axes

This should give the correct rotation of the point in world coordinates. It also gives us the new world -> body transform through the chained matrices.

## Applying Translations and Rotations

The motion of the vehicle will be represented by a translation followed by a rotation. This can be encoded in a spatial transformation matrix, allowing both motions to be calculated simultaneously.