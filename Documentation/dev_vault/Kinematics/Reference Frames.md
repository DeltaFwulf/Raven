
In the simulation, reference frames are all ultimately tied to the world frame, which is tied to a specific location on the Earth's surface. All child frames are simply defined by the transformation required to map from parent to child. In this way, all subcomponents of the vehicle can be located simply by chaining through the vehicle's component tree.

### Vehicle Motion

Vehicle motion is applied in the vehicle's local coordinate system, and consists of a 3d translation and rotation. The 4x4 affine transformation matrix is used to apply these two transformations simultaneously.

Rotations are defined using the quaternion rotation system to avoid gimbal lock. This is convenient in our case, since angular velocity is a vector we can define the motion of the vehicle as an axis of rotation and angle of rotation. 















