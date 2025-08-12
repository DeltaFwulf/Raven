
## Reference Frames

There are three key reference frames in the simulation: The inertial reference frame, which is unmoving and the parent to all other frames. Second is the planet rotating frame, centred on the Earth (or any planet) and able to move relative to the parent frame. Third is the vehicle body frame, a child again of the inertial reference frame.

### Transform Class

The Transform class has been written to represent a specific reference frame (relative to a parent frame), such as mapping vectors defined in one coordinate system into either the parent or child frame of reference, as well as moving the reference frame itself. The relationships between these should be specified inside of a tree structure so that the position of all relative objects can be traced relative to any other.

## Inertia Tensor

### Definition, calculating the inertia tensor of a single object about CoM

- definition, what it is used for
- triple integral

### Calculating the inertia tensor from another frame of reference

- rotating the reference axes
- parallel axis theorem (generalised)
- combining the two types of transformation for a generalised movement

### Calculating the inertia tensor of a compound object

- Order of operations
- Changing an object within the compound object

Firstly, all primitive shapes in the compound must be calculated (mass inertia tensor, com, mass, dimensions, etc), as well as the 'root' locations of all objects within the compound. The root frame within a primitive is the location in the primitive body frame that represents the origin of the part e.g. on the conic primitive, it is the centre of the first end (-x side) of the shape. This allows the primitive to be placed neatly behind other shapes without having to subtract the CoM manually.

The root transform is defined from the centre of mass within the primitive during initialisation, so that the position of the primitive can be specified from the root. When a part's properties are calculated within the compound frame, both the compound->root transform and root->CoM transfer need to be chained to find the compound->CoM transform.

Within the primitive are stored three transforms: 
- CoM location in root coordinates
- Properties relative to CoM: I

Though, we may not need to store any with reference to the root in compound within the primitive, as these will likely just be stored within the compound in some sort of list or dict.
The properties of the object within the compound must be transformed into the compund coordinate system (tied to the compound CoM), so we first need  the compound CoM relative to the compound 'root'

Once we have this, we can get all properties relative to it, and we have the same information as within the primitive: 
- CoM location in root coordinates
- Properties relative to compound CoM: I

This means that when we wish to combine compounds, we don't need to recalculate everything - we can bake this compound into a new, named primitive. This is useful for example when you want to make a nosecone out of multiple hollow conic sections, or battery packs etc from multiple rectangular prisms. It keeps part count down and reduces calculations.

The only parts that must be kept separate are those liable to change, for example tanks with fluid that drains will need recalculating at each timestep and so should be kept at the compound / module stage.



## Angular / Linear Velocities

It may be worth storing velocities within a reference frame in future, as this would permit methods within the frame class that find the velocity of an object within a rotating frame, for use with, e.g. atmospheric flight in a rotating planet's atmosphere. These functions have been derived outside of the reference frame, but not integrated (could be fun, could save code)

