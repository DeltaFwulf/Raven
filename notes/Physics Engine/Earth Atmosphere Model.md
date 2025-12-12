The atmosphere has been modelled using the US Standard Atmosphere (1976). The source paper may be found [here](https://ntrs.nasa.gov/api/citations/19770009539/downloads/19770009539.pdf).  This function can return any of the following values between -5 and +1000 km above sea level:
- Temperature, K
- Pressure, N / m^2
- Density, kg/m^3
- Mean Molecular Weight, kg/kmol
- Dynamic Viscosity, Nm/s (only use up to 86 km)

This function has been integrated into the Earth planet class' *getAtmoProperties* method.
# Convergence Study for Continuous Values
Within the calculations for gas molecular weight, pressure, and density above 86 km, a numerical integration must be performed against z. This means that the answer becomes dependent on the resolution of z used within these integrations. To obtain an efficient method of obtaining 'true' values (those that would be obtained with a continuous integration where $\Delta z = 0$), a convergence study was carried out. Both the direct deviation and rms error were recorded between multiple mesh resolutions (each half the spatial step from the last), and trends found. All three variables exhibited RMS deviations directly proportional to the mesh resolution. Following this realisation, an attempt was made to compensate the error directly. 

![[rms vs resolution.png]]


Given a linear relationship between mesh resolution, $\Delta z$, and RMS deviation $\epsilon$, the following can be deduced:

$$\epsilon_c = \sum_{n = 1}^\infty \epsilon_n$$
where $\epsilon_c$ is the RMS error of the initial mesh results from the continuous result. The error can be expressed as the function of $\Delta z$ :

$$\epsilon_n = f(\Delta z) = \epsilon_1 \frac{\Delta z_n}{\Delta z_1}$$
$$\Delta z_n = f(n) = \frac{\Delta z_1}{2^{n-1}}$$
$$\epsilon_n = \frac{\epsilon_1}{2^{n-1}}$$
Where $\epsilon_1$ is defined as the RMS deviation between the second and first results. n begins at 1 for these relations as there can be no difference for n = 0; this will be accounted for in the final summation.
$$\epsilon_c = \sum_{n=1}^{\infty}\frac{\epsilon_1}{2^{n-1}}$$
Or, equivalently:
$$\epsilon_c = \epsilon_1 \sum_{n=0}^\infty \frac{1}{2^n}$$
The converged value for constant halving is 2, giving:
$$\epsilon_c = 2\epsilon_1$$

This implies that the final value for variable $x$ may be calculated as:
$$ x_c = x_0 - 2 \left(x_1 - x_0 \right) = 3x_0 - 2x_1$$

Under the assumption that the proportional error is constant for all points in the array:

$$ p_n = \frac{x_n - x_{n-1}}{x_{n-1} - x_{n-2}} = \frac{1}{2}$$

This is not strictly true; at large mesh resolutions the proportion between successive deviations flucuates about 0.5. As the mesh is refined, the proportion approximates 0.5. For the compensation to be accurate, the starting resolution has been chosen such that the proportion of the first two deviations ies within 1% of 0.5 for all altitudes. It should be noted that this is not the final accuracy of the compensated value; it will be closer than 1% from the continuous value.

![[prop 100 to 1000 km.png]]

For altitudes between 86 and 100 km, the proportion is particularly non uniform and a finer seed resolution is required.

![[86 to 100 km.png]]

| z range (km) | seed resolution (km) |
| ------------ | -------------------- |
| 86 - 100     | 0.00048828125        |
| 100 - 1000   | 0.015625             |

~~Once the continuous values have been found, coarser resolutions may be compensated directly (idk actually how you'd implement this and why you wouldn't just interpolate these values)~~

The error between the two values was then compensated, giving the following curves for pressure, density, and molecular weight from 86 to 1000 km:

![[compensated.png]]
# Performance
The performance of the function was greatly improved by moving as many calculations into fast number array operations as possible, reducing the computation time by roughly a factor of 4.

Obtaining accurate results with the COESA76 function directly still requires a lot of computation, however, leading to excessive compute time. To avoid this, the values obtained from the convergence study were used to build arrays for pressure, density, and mean molecular weight for altitudes between 86 and 1000 km. These will instead by interpolated between, skipping the computation.

By doing so, the atmosphere model can return any value within 1 millisecond for any altitude.

![[interpolation.png]]


