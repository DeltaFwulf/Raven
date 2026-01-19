import numpy as np
from copy import deepcopy
from math import pi
from numpy.linalg import norm

from vectorUtil import cartesian2coords, coords2cartesian, unit, rotateAxisAngle
from referenceFrame import ReferenceFrame

class Planet():

    def __init__(self, x0, az0):
        self.mu = 0.0
        self.r = 0.0
        self.v = np.zeros(3, float)
        self.omega = 0.0
        self.tilt = 0.0
        self.buildFrames(x0, az0)

    def buildFrames(self, x0:np.array=np.zeros(3, float), az0:float=0.0) -> None:
        """Creates the planet's PCR and PCNR frames"""
        self.PCNR = ReferenceFrame(axis=np.array([0,0,1], float), ang=0.0, origin=x0)
        self.PCNR.moveFrame(origin=np.zeros(3, float), axis=np.array([1,0,0], float), ang=self.tilt, frame='local')
        self.PCR = deepcopy(self.PCNR)
        self.PCR.moveFrame(origin=np.zeros(3, float), axis=np.array([0,0,1], float), ang=az0, frame='local')


    def getAtmoProperties(self, z:float, selected:list['str']=['T', 'p', 'rho']) -> dict:
        """Returns selected atmospheric properties as a dictionary"""
        return {'T':0.0, 'p':0.0, 'rho':0.0}
    
    
    def getAcceleration(self, x:np.array, inputFrame:str='parent', outputFrame:str='parent') -> np.array:
        """Calculates the gravitational acceleration vector given a position relative to the planet.
           Inputs and outputs can be specified in one of three frames: parent, pcr, or pcnr."""

        if inputFrame == 'parent':
            xloc = self.PCR.parent2local(x, incTranslation=True)
        elif inputFrame == 'pcr':
            xloc = x
        elif inputFrame == 'pcnr':
            xloc = self.PCR.parent2local(self.PCNR.local2parent(x, incTranslation=True), incTranslation=True)

        aloc = self.mu*xloc / norm(xloc)**3

        if outputFrame == 'parent':
            aloc = self.PCR.local2parent(aloc, incTranslation=False)
        elif outputFrame == 'pcnr':
            aloc = self.PCNR.parent2local(self.PCR.local2parent(aloc, incTranslation=False))
        
        return aloc
    

    def getZSL(self, x:np.array, inputFrame:str='parent') -> float:
        """Returns altitude above mean sea level"""

        if inputFrame == 'parent':
            xloc = self.PCR.parent2local(x, incTranslation=True)
        elif inputFrame == 'pcr':
            xloc = x
        elif inputFrame == 'pcnr':
            xloc = self.PCR.parent2local(self.PCNR.local2parent(x, incTranslation=False), incTranslation=False)

        return norm(xloc) - self.r
    

    def getCoordinates(self, x:np.array, inputFrame:str='parent') -> tuple['float', 'float']:
        """Given a relative location to the planet, returns PCR latitude and longitude"""

        if inputFrame == 'parent':
            xloc = self.PCR.parent2local(x, incTranslation=True)
        elif inputFrame == 'pcr':
            xloc = x
        elif inputFrame == 'pcnr':
            xloc = self.PCR.parent2local(self.PCNR.local2parent(x, incTranslation=False), incTranslation=False)

        lat, long = cartesian2coords(xloc)

        return lat, long
    

    def xFromCoordinates(self, lat:float, long:float, r:float, outputFrame:str='parent') -> np.array:
        """Returns the position vector that gives an object the specified coordinates on the planet"""

        x = coords2cartesian(lat, long, r)

        if outputFrame == 'parent':
            x = self.PCR.local2parent(x, incTranslation=True)
        elif outputFrame == 'pcnr':
            x = self.PCNR.parent2local(self.PCR.local2parent(x, incTranslation=False), incTranslation=False)
    
        return x

    def getPointingVector(self, x:np.array, pitch:float, heading:float, inputFrame:str='parent', outputFrame:str='parent') -> np.array:
        """Returns a unit vector with desired pitch and compass heading"""

        if inputFrame == 'parent':
            xloc = self.PCR.parent2local(x, incTranslation=True)
        elif inputFrame == 'pcr' or inputFrame == 'pcnr':
            xloc = x
     
        up = xloc # XXX: this assumes spherical planet, must be changed when this is no longer true
        east = norm(up, np.array([0,0,1], float))
        p = rotateAxisAngle(east, up, (pi / 2 - heading) % (2*pi)) # apply heading
        a = np.cross(p, up) # pitch rotation axis
        p = unit(rotateAxisAngle(p, a, pitch)) # apply pitch

        return p
    

    def getLocalVelocity(self, x:np.array, v:np.array, inputFrame:str='parent', outputFrame:str='parent') -> np.array:
        """Returns the relative velocity between an object and a stationary point with same position within the PCR frame
           Note that if the local point's velocity is required, feeding in a velocity of 0 achieves this."""
        
        if inputFrame == 'parent':
            v_pcnr = self.PCNR.parent2local(v - self.v, incTranslation=False)
            x_pcnr = self.PCNR.parent2local(x, incTranslation=True)
        elif inputFrame == 'pcr':
            return v
        elif inputFrame == 'pcnr':
            v_pcnr = v
            x_pcnr = x

        vRel = np.cross(self.omega*np.array([0,0,1], float), x_pcnr) - v_pcnr
 
        if outputFrame == 'parent':
            vRel = self.PCNR.local2parent(vRel, incTranslation=False)
        elif outputFrame == 'local':
            vRel = self.PCR.parent2local(self.PCNR.local2parent(vRel, incTranslation=False), incTranslation=False)

        return vRel



class Earth(Planet):

    def __init__(self, x0, az0):
        self.mu = 3.986e14
        self.r = 6378e3
        self.v = np.zeros(3, float)
        self.omega = 7.292115024e-5
        self.tilt = 0.4090926295
        self.buildFrames(x0, az0)


    # def getAtmoProperties(self, z:float, props:list['str']=['T', 'p', 'rho']) -> dict:
    #     return coesa76(z_m=z, outputs=props, mode='quick')
    


def testMethods():
    earth = Earth(np.zeros(3, float), az0=pi/2)

    # place the rocket on the planet to satisfy coordinates:
    lat = pi / 4
    long = pi / 4
    r = earth.r

    rocketFrame = 'parent'
    outputFrame= 'parent'

    x = earth.xFromCoordinates(lat, long, r, outputFrame=rocketFrame)
    v = np.array([0, 0, 0], float)

    altitude = earth.getZSL(x, inputFrame=rocketFrame)
    lat, long = earth.getCoordinates(x, inputFrame=rocketFrame)
    acceleration = earth.getAcceleration(x, inputFrame=rocketFrame, outputFrame=outputFrame)
    airspeed = earth.getLocalVelocity(x, v, inputFrame=rocketFrame, outputFrame=outputFrame)

    print(f"Rocket frame: {rocketFrame}, Working frame: {outputFrame}")
    print(f"Altitude: {'%.3f' % altitude}")
    print(f"latitude: {'%.3f' % (lat * 180 / pi)} deg, longitude: {'%.3f' % (long * 180 / pi)} deg")
    print(f"Gravitational Acceleration: {'%.3f' % norm(acceleration)} m/s^2")
    print(f"Airspeed: {'%.3f' % norm(airspeed)} m/s")