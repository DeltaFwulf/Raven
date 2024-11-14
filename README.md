# Flight-Simulation

 Full 3D modular spacecraft simulation.

 **Key Features**
 - Modular spacecraft construction with hierarchy
 - Full 6DOF motion simulation
 - Barrowman aerodynamics model
 - Orbital simulation for Earth-Moon system (planet creation suite to be added later)
 - Simple UI for validating rocket construction / visualising flight
 - Plotting functions for trajectory
 - Motor design suite, thrust curves simulated according to engine parameters and feed simulation
 - Emulation of control systems for active guidance simulation, orbital launch optimisation, etc.
 - Flight save and TOML files for quick analysis of previous simulations


## Rocket Definition

**Primitives**
The rocket is, at the lowest level, composed of primitives. These primitives obey rigid body physics. The primitive is defined by its material, shape definition, origin point within a module and a rotation about this origin point. The origin point is usually at the root end of the shape (for a shape aligned to rocket coordinates, the origin will lie along the x principal axis at the +x end of the primitive).

**Modules**  
Modules are equivalent to a rocket's subsystems. They can perform different tasks during flight, and are stored within a stage aboard the rocket. An example of a module would be the rocket motor, which can produce thrust depending on its definition, propellants, and the propellant flow dictated by connected tank physics. To calculate the inertia tensors of each module, one or more primitives are combined at different positions and orientations relative to the module's origin point (at its centre and +x side). Modules are aligned with the rocket's coordinates system. To specify the module type, different classes are inherited to get relevant functions. 

**Stages**  
Stages are the largest component that can be specified in the rocket. These (usually) have their own propulsion system and modules, and can be separated from the vehicle in flight. They are aligned with the rocket coordinate system and are linked to one another either in series or in parallel. 
