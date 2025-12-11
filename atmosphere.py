import numpy as np
from math import exp, sqrt
import matplotlib.pyplot as plt
from time import time



def coesa76(z_m:float, **kwargs) -> dict:
    """
    Calculates Earth's atmospheric properties for a given altitude in metres according to the US 1976 standard atmosphere (COESA76). The desired return values may be specified using a list.

    This list may contain:
    - 'T' for Temperature (K)
    - 'p' for pressure (Pa)
    - 'rho' for density (kg/m^3)
    - 'mw' for mean molecular weight (kg / kmol)
    - 'mu' for dynamic viscosity (Nm/s)

    If the user does not specify any outputs by omitting this argument, all outputs will be returned.
    """

    outs = {'T':True, 'p':True, 'rho':True, 'mw':True, 'mu':True}
    selected = kwargs.get('outputs')
    if selected is not None:
        for key in outs:
            outs[key] = key in selected

    props = {}
    
    z_km = z_m / 1000
    r0_km = 6356.766
    g0 = 9.80665 # m/s^2
    R = 8.31432 # kg / kMol
    mw_0 = 28.9644 # kg / kMol
    kBoltzmann = 1.380662e-23
    kAvogadro = 6.022169e26

    H_arr = [0.0, 11.0, 20.0, 32.0, 47.0, 51.0, 71.0, 84.8520]
    Tm_arr = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
    Lm_arr = [-6.5, 0.0, 1.0, 2.8, 0.0, -2.8, -2.0]
    Hm_arr = [79.0, 79.5, 80.0, 80.5, 81.0, 81.5, 82.0, 82.5, 83.0, 83.5, 84.0, 84.5, 84.852]
    m_arr = [1.0, 0.999996, 0.999988, 0.999969, 0.999938, 0.999904, 0.999864, 0.999822, 0.999778, 0.999731, 0.999681, 0.999679, 0.999579]


    def getTemp(z:float, **kwarg) -> float:
        """Returns temperature wrt z"""
        H = (r0_km*z) / (r0_km + z)
        
        if z < 86:
            Tm = np.interp(H, H_arr, Tm_arr)
            
            if kwarg.get('tm') == True:
                return Tm
            
            else:
                m = np.interp(H, Hm_arr, m_arr)
                T = Tm / m
                return T
        
        else:
            if z < 91:            
                T = 186.8673
                
            elif z < 110:
                T = 263.1905 + -76.3232*sqrt(1 - (z - 91) / -19.9429)

            elif z < 120:
                T = 240 + 12.0*(z - 110.0)
                
            elif z <= 1000:
                T = 1000 - 640*exp(-0.01875*(z - 120.0)*(r0_km + 120.0) / (r0_km + z))
            
            else:
                T = 1000

            return T
    
    
    def getL(z:float) -> float:
        """Returns the first derivative of temperature wrt z"""
        H = (r0_km*z) / (r0_km + z)
        
        if z < 86:
            b = np.max(np.where(H_arr < H))
            L = Lm_arr[b]

        elif z >= 86 and z < 91:
            L = 0.0
        
        elif z < 110:
            A = -76.3232
            a = -19.9429
            z8 = 91
            L = -(A / a) * ((z - z8) / a)*sqrt((1 - (z - z8) / a)**2)
        
        elif z < 120:
            L = 12

        elif z < 1000:
            lamda = 0.01875
            epsilon = (z - 120)*(r0_km + 120) / (r0_km + z)
            L = lamda*640*((r0_km + 120) / (r0_km + z))**2 *exp(-lamda*epsilon)
        
        else:
            L = 0.0

        return L
    

    def getK(z:float) -> float:
        """Returns the eddy diffusion coefficient as a function of z"""
        if 86 <= z < 95:
            K = 120
        elif 95 <= z < 115:
            K = 120*exp(1 - 400 / (400 - (z - 95)**2))
        elif 115 <= z < 1000:
            K = 0.0

        return K
    
    T = getTemp(z_km)

    if outs['T']:
        props.update({'T':T})
    
    if outs['mu']:
        mu = 1.458e-6 * T**1.5 / (T + 110.4)
        props.update({'mu':mu})

    if (outs['p'] or outs['rho']) and z_km < 86:

        p_arr = [101325.0, 22632.06397, 5474.888670, 868.018685, 110.906306, 66.938873, 3.956420, 0.373384]

        H = (r0_km*z_km) / (r0_km + z_km)
        b = np.max(np.where(H_arr < H)) if H > 0 else 0
        
        if Lm_arr[b] != 0.0:
            p = p_arr[b]*(Tm_arr[b] / (Tm_arr[b] + Lm_arr[b]*(H - H_arr[b])))**(g0*mw_0 / (R*Lm_arr[b]))

        else:
            p = p_arr[b]*exp(-g0*mw_0*(H - H_arr[b]) / (R*Tm_arr[b]))

        if outs['p']:
            props.update({'p':p})

        if outs['rho']:
            rho = p*mw_0 / (1000.0*R*getTemp(z_km, tm=True))
            props.update({'rho':rho})

    elif (outs['p'] or outs['rho']) and z_km >= 86:
   
        if kwargs.get('dz_km') is not None:
            dz_km = kwargs.get('dz_km')
        else:
            dz_km = 0.5
        
        n7 = {'n2':1.129794e20, 'o':8.6e16, 'o2':3.030898e19, 'ar':1.351400e18, 'he':7.5817e10}
        alpha = {'he':-0.4, 'h':-0.25}
        a = {'o':6.986e20, 'o2':4.863e20, 'ar':4.487e20, 'he':1.7e21, 'h':3.305e21}
        b = {'o':0.75, 'o2':0.75, 'ar':0.870, 'he':0.691, 'h':0.5}
        Q = {'o':-5.809644e-4, 'o2':1.366212e-4, 'ar':9.434079e-5, 'he':-2.457369e-4}
        U = {'o':56.90311, 'o2':86.0, 'ar':86.0, 'he':86.0}
        W = {'o':2.70624e-5, 'o2':8.333333e-5, 'ar':8.333333e-5, 'he':6.666667e-4}
        w = {'o':5.008765e-4}
        mw = {'n2':28.0134, 'o':15.9994, 'o2':31.9988, 'ar':39.948, 'he':4.0026, 'h':1.00797} # molecular weight, kg/kmol

        n = {'n2':0.0, 'o':0.0, 'o2':0.0, 'ar':0.0, 'he':0.0, 'h':0.0}

        T7 = 186.8673
        Z = np.arange(86, z_km, dz_km) # used for all except hydrogen
        
        TZ = np.zeros_like(Z, float)
        LZ = np.zeros_like(Z, float)
        KZ = np.zeros_like(Z, float)
        gZ = np.zeros_like(Z, float)

        for i in range(Z.size):
            TZ[i] = getTemp(Z[i])
            LZ[i] = getL(Z[i])
            KZ[i] = getK(Z[i])
            gZ[i] = g0*(r0_km / (r0_km + Z[i]))**2

        # Molecular Nitrogen ##########################################################################################
        integral = 0

        for i in range(Z.size):
            M = mw_0 if Z[i] <= 100 else mw['n2']
            integral += dz_km*M*gZ[i] / (R*TZ[i])
        
        n['n2'] = n7['n2']*(T7 / T)*exp(-integral)

        # Atomic Oxygen ###############################################################################################
        M = mw['n2']
        integral = 0
        D = a['o'] / n['n2']*(TZ / 273.15)**b['o']

        for i in range(Z.size):
            fz = gZ[i] / (R*TZ[i]) * (D[i] / (D[i] + KZ[i])) * (mw['o'] + M*KZ[i] / D[i])
            term2 = (-3.416248e-3)*(97 - Z[i])**2 * exp(-w['o']*(97 - Z[i])**3) if Z[i] < 97 else 0
            fv = Q['o']*(Z[i] - U['o'])**2 * exp(-W['o']*(Z[i] - U['o'])**3) + term2
            integral += (fz + fv)*dz_km
            
        n['o'] = n7['o']*(T7 / T)*exp(-integral)

        # Molecular Oxygen ############################################################################################
        integral = 0
        D = a['o2'] / n['n2']*(TZ / 273.15)**b['o2']

        for i in range(Z.size):
            fz = gZ[i] / (R*TZ[i]) * (D[i] / (D[i] + KZ[i]))*(mw['o2'] + M*KZ[i] / D[i])
            fv = Q['o2']*(Z[i] - U['o2'])**2 * exp(-W['o2']*(Z[i] - U['o2'])**3)
            integral += (fz + fv)*dz_km

        n['o2'] = n7['o2']*(T7 / T)*exp(-integral)

        # Argon #######################################################################################################
        integral = 0
        ntot = n['n2'] + n['o'] + n['o2']
        D = a['ar'] / ntot*(TZ / 273.15)**b['ar']
        
        mw_mean = 0
        for key in ['n2', 'o', 'o2']:
            mw_mean += n[key]*mw[key] / ntot

        for i in range(Z.size):
            M = mw_0 if Z[i] < 100 else mw_mean
            fz = (gZ[i] / (R*TZ[i]))*(D[i] / (D[i] + KZ[i]))*(mw['ar'] + M*KZ[i] / D[i])
            fv = Q['ar']*(Z[i] - U['ar'])**2 * exp(-W['ar']*(Z[i] - U['ar'])**3)
            integral += (fz + fv)*dz_km
            
        n['ar'] = n7['ar']*(T7 / T)*exp(-integral)

        # Helium ######################################################################################################
        integral = 0
        D = a['he'] / ntot*(TZ / 273.15)**b['he']

        for i in range(Z.size):
            M = mw_0 if Z[i] < 100 else mw_mean
            fz = (gZ[i] / (R*TZ[i]))*(D[i] / (D[i] + KZ[i]))*(mw['he'] + M*KZ[i] / D[i] + alpha['he']*R*LZ[i] / gZ[i])
            fv = Q['he']*(Z[i] - U['he'])**2 * exp(-W['he']*(Z[i] - U['he'])**3)
            integral += (fz + fv)*dz_km

        n['he'] = n7['he']*(T7 / T)*exp(-integral)

        # Atomic Hydrogen #############################################################################################
        ntot += n['ar'] + n['he']
        mw_mean = 0
        for key in ['n2', 'o', 'o2', 'ar', 'he']:
            mw_mean += n[key]*mw[key] / ntot
        
        if z_km <= 150:
            n['h'] = 0

        else:
            nh11 = 8e10
            phi = 7.2e11
            T11 = 999.2356
            z11 = 500

            Z = np.arange(z_km, z11, dz_km)
            TZ = np.zeros_like(Z, float)
            gZ = np.zeros_like(Z, float)

            for i in range(Z.size):
                TZ[i] = getTemp(Z[i])
                gZ[i] = g0*(r0_km / (r0_km + Z[i]))**2

            integral = 0

            if z_km < z11:
                D = a['h'] / ntot*(TZ / 273.15)**b['h']

                for i in range(Z.size):
                    tau = (mw['h']*dz_km / R)*np.sum(gZ[:i+1] / TZ[:i+1])
                    integral += phi / D[i] * (TZ[i] / T11)**(1 + alpha['h']) * exp(tau) * dz_km

            else:
                tau = (mw['h']*dz_km / R)*np.sum(gZ[:i+1] / TZ[:i+1])

            n['h'] = (nh11 - integral)*(T11 / T)**(1 + alpha['h'])*exp(-tau)

        # Density and Pressure ########################################################################################          
        numDensity = sum([n[key] for key in n])

        if outs['rho']:
            props.update({'rho':sum([n[key]*mw[key] / kAvogadro for key in n])})

        if outs['p']:
            props.update({'p':kBoltzmann*numDensity*T})

    return props



def atmoTest():
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


atmoTest()
#fitProps()