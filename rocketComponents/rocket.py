# rocket class, use this to create a fully specified vehicle.


class Rocket:

    # TODO: figure out a good reference frame (how do we calculate altitude?)
    # - we could define launch site as location on earth (lat,long,el)
    # - initial launch location oriented relative to 'up'?

    # allow arbitrary initial states
    def __init__(self, pos0, vel0, acc0, ang0, angVel0, angAcc0) -> None:
        
        self.stages = []

        self.position = pos0
        self.linVelocity = vel0
        self.linAcceleration = acc0

        self.orientation = ang0
        self.angVelocity = angVel0
        self.angAcceleration = angAcc0


    def addStage(self, stage):
        self.stages.append(stage)

    # internal state methods
    # - get rocket moments of inertia (recursive method over length of vehicle) - write up a spec explaining this
    # - get rocket mass (recursive sum)
    # - get rocket centre of mass (maybe get at the same time as the mass by weighted mean)




class Stage:

    # consists of modules that can be added and removed
    # modules are added at specific locations on the stage

    def __init__(self, name):

        self.modules = [] # NOTE: module order does not matter (perhaps use a map if we need functions to access specific modules)
        self.name = name # use this to denote if the stage is for example, a booster, or payload, or sustainer


    def __str__(self):
        return(f"{self.name}")










