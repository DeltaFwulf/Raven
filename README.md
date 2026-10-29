# Raven Flight Simulator

 A fully three-dimensional flight simulation and mission planner, based upon the principle of modular vehicle definition. Multistage rockets may be constructed and simulated through different scenarios, from simple model rocket launches to fully guided orbital launches (and maybe landing in future..?)

 **Key Features**
 - Modular spacecraft construction with hierarchy
 - Full 6DOF motion simulation
 - Extended Barrowman aerodynamics model
 - n-body gravity model
 - Plotting functions for trajectory
 - Emulation of control systems for active guidance simulation, orbital launch optimisation, etc.
 - Flight save and TOML files for quick analysis of previous simulations


**Required Libraries for Developers**
- numpy 1.26.x
- matplotlib
- pyvista
- numpy-stl


## Rocket Definition

**Modules**
Modules are generally made up from a collection of primitives. The purpose of components is to provide functionality within stages. Modules are to be defined as parametric classes with methods. For example, the 'solid rocket motor' module can provide a thrust vector which can be oriented wrt to the module frame by accepting control inputs at each timestep. The 'fixed-fin' object may be simpler, building itself to a desired contour and providing aerodynamic information to be used within barrowman (or other aerodynamic force function) calculations.

**Primitives**
The rocket is, at the lowest level, composed of primitives. These primitives obey rigid body physics. The primitive is defined by its shape definition and material to give its inertial behaviour.


## Reference Frames
The Universal frame is the parent to all frames, therefore with trivial rotation and translation. All frames are right handed.

**Rockets**
- Root Frame (root_frame): tied to the nose of the rocket, with +x pointed away from the spicy end. Expressed wrt to the universal frame.
- CoM Frame (com_frame): aligned with the root frame, but with origin at the rocket's time-dependent centre of mass. Expressed wrt to the root frame.
- Stages all have their own root and com frames, as do modules and primitives. These have the same heirarchy i.e. roots are expressed wrt to direct parent frame, and com are expressed wrt to their peer root frame.

**Planets**
- Planet-Centered Rotating (PCR): moves such that objects stationary to the surface of the planet are stationary in the frame. z aligned to axis of rotation, x aligned to both equator and prime meridian, y normal to both.
- Planet-Centered Non-Rotating (PCNR): Origin at planet barycentre, z aligned to axis of rotation, x aligned to vernal equinox direction (also +x in universal frame), and y normal to both.
Both planet frames are expressed wrt the universal frame.






