import numpy as np
from copy import deepcopy

from vectorUtil import ReferenceFrame
from atmosphere import coesa76

class Planet():

    def __init__(self, x0, az0):
        self.mu = 0.0
        self.r = 0.0
        self.omega = 0.0
        self.tilt = 0.0
        self.buildFrames(x0, az0)

    def buildFrames(self, x0:np.array=np.zeros(3, float), az0:float=0.0) -> None:
        """Creates the planet's PCR and PCNR frames"""
        self.PCNR = ReferenceFrame(axis=np.array([0,0,1], float), ang=0.0, translation=x0)
        self.PCNR.moveAbout(origin=np.zeros(3, float), axis=np.array([1,0,0], float), ang=self.tilt, frame='local')
        self.PCR = deepcopy(self.PCNR)
        self.PCR.moveAbout(origin=np.zeros(3, float), axis=np.array([0,0,1], float), ang=az0, frame='local')


    def getAtmoProperties(self, z:float, selected:list['str']=['T', 'p', 'rho']) -> dict:
        """Returns selected atmospheric properties as a dictionary"""
        return {'T':0.0, 'p':0.0, 'rho':0.0}
    

    
class Earth(Planet):

    def __init__(self, x0, az0):
        self.mu = 3.986e14
        self.r = 6378e3
        self.omega = 7.292115024e-5
        self.tilt = 0.4090926295
        self.buildFrames(x0, az0)


    def getAtmoProperties(self, z:float, props:list['str']=['T', 'p', 'rho']) -> dict:
        return coesa76(z_m=z, outputs=props)