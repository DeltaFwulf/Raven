# Objectives

- overall objective
	- near term scope
	- long term goals (aids how code is set up)


A top down approach will be taken to the code, rather than the previously attempted bottom up approach which led to a lot of parts with no obvious path to implementation in a simulation.

# Software Scope (Phase I)
- text based simulation script with a single body launch (unguided) from a position on a planet.
- plots and outputs of altitude, orientation, accelerations etc.

# Simulation Requirements (MVP)
- 3d motion captured for a rocket
- aerodynamic forces using extended barrowman
- RK4 in all explicit simulations
- unguided launch (model rocket or sounding rocket)
- guided launch to orbit using UPFG, with steering commands
- flight trajectory results and plotting suite
# Rocket Requirements and Definition
Flights are simulated for a single rocket. This rocket may consist of one or more *Stages*, which may be detached throughout flight. These staging events create separate objects to track, but which perform no actions as of yet. The rocket's stages are made up from multiple parametric *components* which can perform actions or not. For example, a simple component would be a body tube or interstage, which acts as an inert object, but the solid rocket motor produces thrust and can be vectored to produce control moments on the vehicle. All components are made up of one or more *primitives*, which are parametric shapes, made from defined materials, and that act as inertial objects (with moments of inertia, centres of mass, etc). By placing the components together into stages, then stacking stages as desired, the vehicle is defined. 

The rocket must be able to be saved to a plain-text file format such as .toml, and loaded within the simulation software.

Some visualiser must be developed to show the vehicle after it has been created / loaded so that the user can check it looks correct.

# Planet Requirements and Definition
There must be at least one planet from which to launch. This planet is positioned relative to the universal frame, with axial tilt, initial azimuth, angular and linear velocities, and two frames: Planet-Centered Rotating and Planet-Centered Nonrotating frames, both having origins at the planet barycentre but in the rotating frame, objects stationary to the surface of the planet are stationary in the frame. The other is defined by vernal equinox, axis of rotation, and the right hand normal unit vector.

The planet will have an atmosphere that returns gas properties such as temperature, pressure, and density as a function of altitude. If a standardised atmospheric model is available and is lightweight enough to run within a simulation, this will be implemented (for example, for Earth, the COESA76 model will be used).

The planet may have non-spherical gravity if given the appropriate weights. The field strength in this case is a function of spherical coordinates.

The initial planet scope is only for spherical planets with spherical gravity.
# Physics Engine Requirements

The physics engine must calculate the motion of bodies, capturing both rotation and translation within 3d space. Motion must capture multibody systems with variable mass.

Accelerations may come from aerodynamic forces, thrust, or gravity.

Equations of motion  must be solvable using RK4 and should be expressable in vector form to make calculations simpler.