Primitives are the building blocks of modules, and allow the vehicle's kinematics to be determined in a detailed and modular fashion. Primitives have a shape, material, and position relative to its parent module's origin. 

## Primitive Species
[[Conic Full]]
[[Conic Generalised]]
[[Rectangular Prism]]
[[Custom Primitive]]

### To Be Defined:
planar fins
common nosecone profiles

## Spatial Position

All of a primitive's locations are tied to its origin point. Primitives are generally centred about the +x axis in this frame, with their root at (0, 0, 0). A primitive's coordinate system is generally the same as the part's principal axes, however not necessarily (for example, if a custom, asymmetric part is defined). We assume that all information about a primitive is relative to this origin point and in the part's reference frame.

^ Do we want to do this? What we really want is the part's properties in the module reference frame. Therefore, we should calculate the part relative to its axes but transform the outputs into module coordinates before returning values.

When creating a primitive, we must specify its spatial position with respect to its parent module's origin. We provide both a translation and rotation between the module origin and the part origin. 


There are two methods I'm considering to transform my vectors: spatial transformation matrix and quaternions + separate translations. 
- Due to several advantages of using Quaternions, I will select this method first, and take it as far as I can before I get lost. I believe that I can choose rotation order by simply rotating in 3 directions..? \


We should be able to find the angular momentum's axis of rotation by vectorially adding together all of the different angular momenta. 

