Primitives are the building blocks of modules, and allow the vehicle's kinematics to be determined in a detailed and modular fashion. Primitives have a shape, material, and position relative to its parent module's origin. 

## Primitive Species


### To Be Defined:

Laminar fins

[Inertia Tensor MIT](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/dd277ec654440f4c2b5b07d6c286c3fd_MIT16_07F09_Lec26.pdf)
[3D Kinematics MIT](https://ocw.mit.edu/courses/16-07-dynamics-fall-2009/419be4d742e628d70acfbc5496eab967_MIT16_07F09_Lec25.pdf)

## Intrinsic Properties

All of a primitive's locations are tied to its origin point. Primitives are generally centred about the +x axis in this frame, with their root at (0, 0, 0). A primitive's coordinate system is generally the same as the part's principal axes, however not necessarily (for example, if a custom, asymmetric part is defined). We assume that all information about a primitive is relative to this origin point and in the part's reference frame.

There are two methods I'm considering to transform my vectors: spatial transformation matrix and quaternions + separate translations. 
- Due to several advantages of using Quaternions, I will select this method first, and take it as far as I can before I get lost. I believe that I can choose rotation order by simply rotating in 3 directions..? \

We should be able to find the angular momentum's axis of rotation by vectorially adding together all of the different angular momenta. 


A primtive represents a shape of constant density and can a variant of these species:
[[Conic Full]]
[[Conic Generalised]]
[[Rectangular Prism]]
[[Custom Primitive]]

All primitives have certain intrinsic properties:
- mass
- CoM relative to primitive's root
- MoI about the primitive's CoM
- edges and vertices for openGL

With this information, the part can be used within modules.
