import matplotlib.pyplot as plt
from time import time
from copy import deepcopy
import numpy as np
from math import sqrt
from os import getcwd
from os.path import join

from COESA76 import coesa76

def speedTest():
    """Calculates and plots properties over a range of altitudes. Generates a curve of calculation time for optimisation purposes"""

    z_m = np.arange(0, 1000e3, 1000)
    z_km = z_m / 1000
    dz_km = 0.1
    selection = ['T', 'rho', 'p', 'time']

    arrays = {}
    for var in selection:
        arrays.update({var:np.zeros_like(z_m, float)})

    for i in range(z_m.size):

        t0 = time()
        atmo = coesa76(z_m[i], dz_km=dz_km, outputs=selection)
        atmo.update({'time':1000*(time() - t0)})

        for var in atmo:
            arrays[var][i] = atmo[var]

    if 'T' in selection:

        fig0, axT = plt.subplots()
        colT = 'black'
        fig0.suptitle("Temperature vs Altitude")
        axT.plot(arrays['T'], z_km, '-', color=colT)
        axT.tick_params(axis='x', labelcolor=colT)
        axT.set_xlabel('T, K')
        axT.set_ylabel('z, km')

        fig0.tight_layout()

    if 'p' in selection or 'rho' in selection:

        fig1, ax0 = plt.subplots()
        ax0.set_xscale('log')
        ax0.set_ylabel("Altitude, km")

        colP = 'tab:red'
        colR = 'tab:blue'

        if 'p' in selection: # takes priority
            
            ax0.plot(arrays['p'], z_km, '-', label='pressure', color=colP)
            ax0.tick_params(axis='x', labelcolor=colP)
            ax0.set_xlabel("Pressure, Pa")

            if 'rho' in selection:
                ax1 = ax0.twiny()
                ax1.set_xscale('log')
                ax1.plot(arrays['rho'], z_km, '-', label='density', color=colR)
                ax1.tick_params(axis='x', labelcolor=colR)
                ax1.set_xlabel("Density, kg/m^3")
                fig1.suptitle("Pressure and Density vs Altitude")

            else:
                fig1.suptitle("Pressure vs Altitude")
        
        else:
            ax0.plot(arrays['rho'], z_km, '-', label='density', color=colR)
            ax0.tick_params(axis='x', labelcolor=colR)
            ax0.set_xlabel("Density, kg/m^3")
            fig1.suptitle("Density vs Altitude")
 
    fig1.legend(loc='right', bbox_to_anchor=(0.95, 0.5))
    fig1.tight_layout()

    # show function performance:
    if 'time' in selection:

        colTime = 'grey'

        # rolling mean with specified width
        width = 10
        t = arrays['time']
        ra = np.zeros_like(t, float)

        for i in range(t.size):
            lb = int(max(0, i - width / 2))
            ub = int(min(t.size, i + width / 2))
            ra[i] = np.mean(t[lb:ub])

        # key altitudes to monitor:
        keyZ = {'nd start':86, 'h start':150, 'h high':500}

        figt, axt = plt.subplots()
        figt.suptitle("Calculation Time vs Altitude")
        axt.plot(z_km, arrays['time'], '-', color=colTime, linewidth=1.2, label='dt')
        axt.plot(z_km, ra, '-', color='tab:blue', linewidth=1.5, label=f"rolling mean, width {width}")
        axt.tick_params(axis='y', labelcolor=colTime)
        axt.set_xlabel("Altitude, km")
        axt.set_ylabel("Cycle time, ms")
        axt.grid(True)

        for key in keyZ:
            axt.plot([keyZ[key], keyZ[key]], [0, np.max(arrays['time'])], '--r', linewidth=1.2, label=key)

        axt.legend(loc='lower right')
        figt.tight_layout()

    plt.show()


def convergenceStudy():

    dz_km = 1
    maxDouble = 8

    z_m = np.arange(100e3, 1e6 + 100, 1000)
    z_km = z_m / 1000
    selection = ['p', 'rho', 'mw'] # only these values rely on integral calculations

    out = {}
    last = {}
    dev = {}
    devLast = {}
    cumDev = {}
    rms = {}
    res = []
    
    for var in selection:
        out.update({var:np.zeros_like(z_m, float)})
        last.update({var:np.zeros_like(z_m, float)})
        dev.update({var:np.zeros_like(z_m, float)})
        devLast.update({var:np.zeros_like(z_m, float)})
        cumDev.update({var:np.zeros_like(z_m, float)})
        rms.update({var:[]})

    fig, ax = plt.subplots()
    fig.suptitle("Deviation / Previous Deviation")
    ax.set_xscale('log')
    ax.set_xlabel('Altitude, km')
    ax.set_ylabel('proportion')
    ax.plot([z_km[0], z_km[-1]], [0.5, 0.5], '--g', label='target', linewidth=1.2)
    ax.plot([z_km[0], z_km[-1]], [0.495, 0.495], '--r', label='lower bound', linewidth=1.0)
    ax.plot([z_km[0], z_km[-1]], [0.505, 0.505], '--r', label='upper bound', linewidth=1.0)

    for d in range(0, maxDouble):
        print(f"Entering loop {d+1} of {maxDouble} with resolution = {dz_km} km")

        for var in selection:
            last[var] = deepcopy(out[var])

        for i in range(z_m.size):
            atmo = coesa76(z_m[i], dz_km=dz_km, outputs=selection)

            for var in atmo:
                out[var][i] = atmo[var]
    
        if d > 0:
            res.append(dz_km)
            for var in selection:
                devLast[var] = dev[var]
                dev[var] = (out[var] - last[var]) / last[var]
                cumDev[var] += dev[var]
                rms[var].append(sqrt(np.sum(dev[var]**2)))

            scaled = (dev['p'] / devLast['p'])
            ax.plot(z_km, scaled, '-', label=f"{dz_km}", linewidth=1.2)
        
        dz_km /= 2

    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    fig1, ax1 = plt.subplots()
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    ax1.set_xlabel("Resolution (km)")
    ax1.set_ylabel("RMS deviation")
    fig1.suptitle("RMS deviation vs Integral Resolution, COESA76")
    for var in selection:
        ax1.plot(res, rms[var], '-', label=var)

    ax1.legend()
    fig1.tight_layout()

    plt.show()



def getContinuous():
    """Compensates the predicted error from continuous for pressure, density, and molecular weight over the altitude range of 86 to 1000 km so that arrays may be built and interpolated within coesa76's 'quick' mode"""
    z_low = np.arange(86e3, 100e3, 500)
    z_high = np.arange(100e3, 1000500, 500)

    # generate initial arrays for 86 to 100km
    dz_low = 0.00048828125
    dz_high = 0.015625

    #dz_low = 0.25
    #dz_high = 0.25

    selection = ['p', 'rho', 'mw']

    low = {}
    high = {}
    comp = {}
   
    for var in selection:
        low.update({var:np.zeros((z_low.size, 2), float)})
        high.update({var:np.zeros((z_high.size, 2), float)})
        comp.update({var:np.zeros(z_low.size + z_high.size, float)})
       
    for i in range(2):

        print(f"Starting loop {i+1} of 2")

        for j in range(z_low.size):
            atmo = coesa76(z_low[j], dz_km=dz_low, outputs=selection)

            for var in atmo:
                low[var][j, i] = atmo[var]

        for j in range(z_high.size):
            atmo = coesa76(z_high[j], dz_km=dz_high, outputs=selection)

            for var in atmo:
                high[var][j, i] = atmo[var]

        dz_low /= 2
        dz_high /= 2

    for var in selection:
        comp[var] = 3*np.concatenate((low[var][:,0], high[var][:,0])) -2*np.concatenate((low[var][:,1], high[var][:,1]))

    z_km = np.hstack((z_low, z_high)) / 1000

    # plot results
    fig, ax = plt.subplots()
    fig.suptitle("Pressure and Density vs Altitude, Compensated")
    ax.set_xscale('log')
    ax.set_ylabel("Altitude, km")

    colP = 'tab:red'
    colR = 'tab:blue'

    ax.plot(comp['p'], z_km, '-', label='pressure', color=colP, linewidth=1.5)
    ax.tick_params(axis='x', labelcolor=colP)
    ax.set_xlabel("Pressure, Pa")

    ax1 = ax.twiny()
    ax1.set_xscale('log')
    ax1.plot(comp['rho'], z_km, '-', label='density', color=colR, linewidth=1.5)
    ax1.tick_params(axis='x', labelcolor=colR)
    ax1.set_xlabel("Density, kg/m^3")

    fig.tight_layout()

    fig1, ax1 = plt.subplots()
    fig1.suptitle("Air Mean Molecular Weight vs Altitude, Compensated")
    ax1.plot(comp['mw'], z_km, '-', color='tab:green', label='molecular weight', linewidth=1.5)
    ax1.set_xlabel("Mean Molecular Weight, kg/kmol")
    ax1.set_ylabel("Altitude, km")
    
    fig1.tight_layout()

    path = join(getcwd(), 'coesa76.npz')
    np.savez(path, z_km=z_km, p=comp['p'], rho=comp['rho'], mw=comp['mw'])

    plt.show()
    




def fitProps():
    
    z_m = np.arange(0, 1e6, 1000)
    z_km = z_m / 1000
    dz_km = 0.1
    selection = ['p', 'rho']
    order = 10

    arrays = {}
    for var in selection:
        arrays.update({var:np.zeros_like(z_m, float)})

    for i in range(z_m.size):
        atmo = coesa76(z_m[i], dz_km=dz_km, outputs=selection)
        for var in atmo:
            arrays[var][i] = atmo[var]

    # curve fit
    propArr = np.transpose(np.vstack((arrays['p'], arrays['rho'])))
    res = np.polyfit(x=z_km, y=propArr, deg=order)


    def polynomial(x:np.array, coeffs:np.array) -> np.array:

        y = np.zeros_like(x, float)

        for i in range(x.size):
            for j in range(len(coeffs)):
                y[i] += coeffs[j]*x[i]**(len(coeffs) - j - 1)

        return y


    pFit = polynomial(z_km, res[:,0])
    rhoFit = polynomial(z_km, res[:,1])

    fig, axP = plt.subplots()
    color = 'tab:red'
    axP.set_xscale('log')
    axP.plot(arrays['p'], z_km, '-', label='pressure', color=color)
    axP.plot(pFit, z_km, '--', label='pressure fit', color=color)
    axP.tick_params(axis='x', labelcolor=color)
    axP.set_xlabel("Pressure, Pa")
    axP.set_ylabel("Altitude, km")

    axR = axP.twiny()
    color = 'tab:blue'
    axR.set_xscale('log')
    axR.plot(arrays['rho'], z_km, '-', label='density', color=color)
    axR.plot(rhoFit, z_km, '--', label='density fit', color=color)
    axR.tick_params(axis='x', labelcolor=color)
    axR.set_xlabel("Density, kg/m^3")
    
    fig.suptitle(f"Pressure and Density vs altitude compared to curve fits of order {order}")
    fig.legend()

    plt.show()


speedTest()
#fitProps()
#convergenceStudy()
#getContinuous()