import numpy as np
from math import exp, sqrt



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

    mode = kwargs.get('mode') if kwargs.get('mode') is not None else 'quick'

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
        
        if z < 86:
            H = (r0_km*z) / (r0_km + z)
            Tm = np.interp(H, H_arr, Tm_arr)
            
            if kwarg.get('tm') == True:
                return Tm
            
            else:
                m = np.interp(H, Hm_arr, m_arr)
                T = Tm / m
        
        elif z < 91:            
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
        
        if z < 86:
            H = (r0_km*z) / (r0_km + z)
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
    

    T = getTemp(z_km) # FIXME: this is no longer optimal, mw is independent of T when z_km < 86

    if outs['mw'] and z_km < 86:
        if z_km < 79.9941:
            props.update({'mw':mw_0})
        else:
            H = (r0_km*z_km) / (r0_km + z_km)
            props.update({'mw':mw_0*np.interp(H, Hm_arr, m_arr)})

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

    elif (outs['p'] or outs['rho'] or outs['mw']) and z_km >= 86 and mode == 'quick':
        arrs = np.load('./atmosphere/coesa76_baked.npz')

        if outs['p']:
            props.update({'p': np.interp(z_km, arrs['z_km'], arrs['p'])})

        if outs['rho']:
            props.update({'rho': np.interp(z_km, arrs['z_km'], arrs['rho'])})

        if outs['mw']:
            props.update({'mw': np.interp(z_km, arrs['z_km'], arrs['mw'])})


    elif (outs['p'] or outs['rho'] or outs['mw']) and z_km >= 86 and mode == 'direct':
   
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
        gZ = g0*(r0_km / (r0_km + Z))**2
        
        TZ = np.zeros_like(Z, float)
        LZ = np.zeros_like(Z, float)
        KZ = np.zeros_like(Z, float)
        
        for i in range(Z.size):
            TZ[i] = getTemp(Z[i])
            LZ[i] = getL(Z[i])
            KZ[i] = getK(Z[i])

        # Molecular Nitrogen ##########################################################################################
        M = np.zeros_like(Z, float)
        for i in range(Z.size):
            M[i] = mw_0 if Z[i] < 100 else mw['n2']
        
        integral = (dz_km / R)*np.sum(gZ*M / TZ)
        n['n2'] = n7['n2']*(T7 / T)*exp(-integral)

        # Atomic Oxygen ###############################################################################################
        M = mw['n2']
        D = a['o'] / n['n2']*(TZ / 273.15)**b['o']

        if Z.size == 0 or Z[0] > 97:
            k97 = 0
        else:
            k97 = np.max(np.where(Z < 97))
        
        fz = gZ / (R*TZ)*(D / (D+KZ))*(mw['o'] + M*KZ / D)
        fv = Q['o']*(Z - U['o'])**2 * np.exp(-W['o']*(Z - U['o'])**3)
        fv[:k97] += (-3.416248e-3)*(97 - Z[:k97])**2 * np.exp(-w['o']*(97 - Z[:k97])**3)
        integral = np.sum(fv + fz)*dz_km
        n['o'] = n7['o']*(T7 / T)*exp(-integral)

        # Molecular Oxygen ############################################################################################
        D = a['o2'] / n['n2']*(TZ / 273.15)**b['o2']
        fz = gZ / (R*TZ)*(D / (D + KZ))*(mw['o2'] + M*KZ / D)
        fv = Q['o2']*(Z - U['o2'])**2 * np.exp(-W['o2']*(Z - U['o2'])**3)
        integral = np.sum(fz + fv)*dz_km
        n['o2'] = n7['o2']*(T7 / T)*exp(-integral)

        # Argon #######################################################################################################
        ntot = n['n2'] + n['o'] + n['o2']
        mw_mean = sum([n[key]*mw[key] for key in ['n2', 'o', 'o2']]) / ntot
        D = a['ar'] / ntot*(TZ / 273.15)**b['ar']
        M = np.zeros_like(Z, float)
        for i in range(Z.size):
            M[i] = mw_0 if Z[i] < 100 else mw_mean

        fz = (gZ / (R*TZ))*(D / (D + KZ))*(mw['ar'] + M*KZ / D)
        fv = Q['ar']*(Z - U['ar'])**2 * np.exp(-W['ar']*(Z - U['ar'])**3)
        integral = np.sum(fz + fv)*dz_km
        n['ar'] = n7['ar']*(T7 / T)*exp(-integral)

        # Helium ######################################################################################################
        D = a['he'] / ntot*(TZ / 273.15)**b['he']
        fz = (gZ / (R*TZ))*(D / (D + KZ))*(mw['he'] + M*KZ / D + alpha['he']*R*LZ / gZ)
        fv = Q['he']*(Z - U['he'])**2 * np.exp(-W['he']*(Z - U['he'])**3)
        integral = np.sum(fz + fv)*dz_km
        n['he'] = n7['he']*(T7 / T)*exp(-integral)

        # Atomic Hydrogen #############################################################################################
        ntot += n['ar'] + n['he']
        
        if z_km <= 150:
            n['h'] = 0

        else:
            nh11 = 8e10
            T11 = 999.2356
            z11 = 500

            Z = np.arange(z_km, z11, dz_km) if z_km < z11 else np.arange(z11, z_km, dz_km)
            gZ = g0*(r0_km / (r0_km + Z))**2

            TZ = np.zeros_like(Z, float)
            for i in range(Z.size):
                TZ[i] = getTemp(Z[i])
               
            expTau = np.exp((mw['h']*dz_km / R)*np.cumsum(gZ / TZ)) if Z.size > 0 else np.ones(1, float)
            
            if z_km < z11:
                D = a['h'] / ntot*(TZ / 273.15)**b['h']
                integral = (7.2e11 * dz_km)*np.sum(1 / D * (TZ / T11)**(1 + alpha['h'])*expTau)
                n['h'] = (nh11 - integral)*(T11 / T)**(1 + alpha['h']) / expTau[-1]

            else:
                n['h'] = nh11*(T11 / T)**(1 + alpha['h']) / expTau[-1]
            
        ntot += n['h']
       
        if outs['rho']:
            props.update({'rho':sum([n[key]*mw[key] / kAvogadro for key in n])})

        if outs['p']:
            props.update({'p':kBoltzmann*ntot*T})

        if outs['mw']:
            props.update({'mw':sum([n[key]*mw[key] for key in ['n2', 'o', 'o2', 'ar', 'he']]) / ntot})

    return props