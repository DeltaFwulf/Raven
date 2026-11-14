"""Plot atmospheric properties against z, 0 to 1000 km."""

import numpy as np
import matplotlib.pyplot as plt

from raven.atmosphere import coesa76

def calc_state_coesa() -> None:
    """Plot temperature, pressure, density vs altitude between 0 and 1000 km altitude.
    
    state is calculated via the coesa76 atmosphere."""

    alts = np.arange(0, 1e6, 1000.0)
    props = np.zeros((alts.size, 3), float)

    for i, z in enumerate(alts):
        props[i, :] = coesa76(z)


    # plot results
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(alts / 1000.0, props[:, 0], '-r')
    axs[0].set_ylabel("T, K")
    axs[0].grid(True)

    axs[1].plot(alts / 1000.0, props[:, 1], '-b')
    axs[1].set_ylabel("p, Pa")
    axs[1].grid(True)

    axs[2].plot(alts / 1000.0, props[:, 2], '-g')
    axs[2].set_ylabel("$\\rho$, kg/m^3")
    axs[2].grid(True)

    fig.supxlabel("z, km")
    plt.tight_layout()
    plt.show()


calc_state_coesa()