It would be more convenient to pass all information about the vehicle's state into a single function, than updating linear, angular, and multibody information separately. 

To do this, we will build a state space vector:

- x
- y
- z
- u
- v
- w
- q0
- q1
- q2
- q3
- omegax
- omegay
- omegaz

The update function will take in this state vector as well as several solver settings:
- linear mode
- angular mode
- timestep
- any constants

