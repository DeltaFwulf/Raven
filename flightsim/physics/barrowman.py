# Calculates rocket stability and forces given an angle of attack from the freestream.
# (This can be mapped into the correct reference frame later down the line as a torque pseudovector)

from math import pi, sqrt, cos, sin
import numpy as np

from atmosphere import isa_atmo

# TODO: get the correct freestream angle for summing a rolled vehicle angled to the freestream


def vFlutter(cr, ct, h, tMin, z):

    # TODO: allow material properties to be parameterised

    # Calculates the flutter speed of a fin
    R = 287
    gamma = 1.4

    AR = h / (0.5 * (cr + ct))
    T, P = isa_atmo(z)[:-1]
    P0 = isa_atmo(0)[1]
    
    pRatio = P / P0
    Y = 24 * 0.25 * gamma * P0 / pi
    a = sqrt(gamma * R * T)

    G = 30e9 / 2 # NOTE: this may well be inaccurate, due to anisotropy

    k1 = Y * AR**3 / ((tMin/cr)**3 * (AR + 2))
    k2 = (ct / cr + 1) / 2

    vf = a * sqrt(G / (k1 * k2 * pRatio))  
    print(f"Predicted Flutter Speed: {vf} m/s")

    return vf  



def splitfin(pointsCCW:np.array):
    # If a fin profile is not of the form compatible with barrowman analysis, it can be split into a series of fins that can.
    pass



def bmFin(cr:float, ct:float, h:float, sweepAng:float, dRef:float, M:float=0, plot:bool=False):
    # Gets the aerodynamic properties of a barrowman compatible fin
    n = 0.25 # changes with Mach number
    l = h / cos(sweepAng)
    
    cnAlpha = 8 * (h / dRef)**2 / (1 + sqrt(1 + (2 * l / (cr + ct))**2))

    # Centre of pressure (x, y)    
    xt = l * sin(sweepAng) - (n * ct)

    y = (h/3) * (2*cr + ct) / (cr + ct)
    x = (xt / 3) * (cr + 2 * ct) / (cr + ct) + n * 2/3 * (cr + ct - (cr*ct) / (cr + ct))

    return cnAlpha, x, y



def bmConic(d0:float, d1:float, l:float, dRef:float):
    
    cnAlpha = 8 / (pi * dRef**2) * (d1 - d0)

    if cnAlpha == 0:
        x = 0 # it's a cylinder, so x is undefined and irrelevant

    else:
        x = l/3 * (1 + 1/(1 + d0 / d1))


def bmNosecone(dBase:float, l:float, type:str, arg:float=0):

    cnAlpha = 2

    match type:

        case 'conic':
            vol = pi * dBase ** 2 / 12

        case 'haack':
            vol = 0


    x = l - vol / (pi * dBase**2 / 4)



def getCP():

    nosecone = Nosecone(d0=50, l=150, type='haack', arg=0.25)





vFlutter(cr=80e-3, ct=30e-3, h=70e-3, tMin=2.5e-3, z=0)


