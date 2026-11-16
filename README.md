# Raven Flight Simulator

A spaceflight simulation suite, with plans to include launch ascent guidance and optimisation, mission planning, and digital twin capability. I'm making this as the tool I want to simulate anything from an L1 model rocket to an interplanetary mission with landing. This is a very slow burn project, and is likely not as user friendly or polished enough for general use (help is very much appreciated!) but it can at least serve as a repository of simulation example code if anyone wants to take a look through.

I plan to develop this simulation with a modular rocket definition, so parts can be added, adjusted, etc. This may be useful i.e. to investigate how important certain tolerances are (like nozzle or fin alignment on small rockets), or optimising stage proportions for multistage rockets (at a first-pass level, of course). Modules will have their own state and can, for example, emulate a microcontroller or connect via serial port and provide HIL capability for a flight computer. I plan to use this in my own amateur rocketry hobby, and think that it'd be super cool to validate my code before flight with my own sims.

<img width="1017" height="761" alt="free_return" src="https://github.com/user-attachments/assets/d581dc08-abe3-431b-8f33-19c39ad91569" />

$\uparrow$ _A lunar free return trajectory simulated with Raven._
<br>

<br>

 **Planned Features**
 - Modular spacecraft, accepts .stl meshes for component for CAD integration.
 - Programmable modules for dynamic vehicle behaviours (nozzle gimbal, fin actuation, reaction wheels etc).
 - Extended Barrowman aerodynamics model.
 - n-body gravity model with Runge-Kutta-Fehlberg solver.
 - Ascent guidance and trajectory optimisation.
 - Mission planner to reach target trajectories.
 - HIL via serial port, connection to ground station via TCP.

<br>

<br>

**Dependencies**
- numpy
- pyvista
- pyyaml

I will try my best to add my growing list of notes to the wiki, to explain my thought process during development. If you have any questions (for some reason), please send me an email or add an issue.
