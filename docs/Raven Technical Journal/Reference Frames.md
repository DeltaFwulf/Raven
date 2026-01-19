Reference frames are sets of orthogonal axes in which vectors may be expressed, for example, positions and velocities only exist within some reference frame such at the planet's PCNR frame. Raven requires a ReferenceFrame class to store and modify reference frame spatial information, as well as functions used to transform vectors into and out of specific frames of reference.

A reference frame is defined by its relative translation and rotation to a parent frame (any frame must exist within a higher level frame). If you're wondering how to break this infinite chain of heirarchies, simply make one frame identical to its parent and treat it as the 'universal' frame.

Reference frames use quaternions for rotation to avoid gimbal lock and reduce computation.

# Required Features

## Defining Initial Position
When first defining the position of a reference frame, some information is required about translation and rotation. The translation component is simple; this is represented by a vector in the parent frame. For rotation, multiple forms may be best within the simulation, with two methods, defined below, included at this stage of development. It is important that the function raises appropriate exceptions with useful messages if the wrong rotation configuration is sent to the init function.

**Vectors**
One or two vectors may be passed in, with a list of strings specifying the sequence of vector alignments (made from either 'x', 'y', 'z'). An axis angle rotation will be performed to align the vector, but roll must be controlled later if required. Optionally, a second vector can be specified with its name. This vector does not have to be normal to the original but cannot be parallel; it will be projected normal to the first and made a unit vector. The two rotations required are computed then quaternion obtained. The second vector is useful if some 'up' vector or something is known and the +z or whatever axis should face it as closely as possible; this is not required.

**Axis Angle**
Directly obtains the quaternion representation of this axis angle rotation for the frame. Requires parameter 'axis' unit vector and 'angle' float.

## Moving the Reference Frame

When objects in the simulation move, the reference frames representing their position must be moved correspondingly. Two types of movement may be applied: rotation and translation. Rockets' positions are to be expressed in terms of some 'root' frame, which sits at the +x centre of the vehicle. When the rocket rotates in free flight, it does so about its centre of mass rather than the root frame; it is therefore advantageous to permit rotation about some arbitrary origin relative to the frame's current location. In this example, while it may be simpler to express centre of mass in local coordinates, the vehicle moves through space with velocities expressed in universal coordinates; it would therefore be useful to express movements in terms of parent coordinates or local independently from the origin working frame.

### Derived Requirements
- frames can be moved by translations and / or rotations
- frames can be rotated about an arbitrary origin relative to the frame's origin
- the origin and movements may be expressed in local or parent coordinates
- the working frame of the movement is indpenedent of the origin working frame
- translations may be specified without any rotation, and vice versa, without explicitly giving trivial rotations or translations as arguments
- if no origin is given, it is set to be a vector with coordinates (0i, 0j, 0k) in the origin working frame
- translations are represented by a numpy ndarray
- rotations are represented by an axis and angle (numpy ndarray and float)
- origins are represented by a numpy ndarray
- the origin default working frame is local to the frame's axes
- the movement default working frame is the parent frame
- working frames for both origins and movements are represented by the standard strings, 'parent' or 'local'
- the function has no explicit return values

## Vector Transformation


