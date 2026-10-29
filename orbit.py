"""Orbit determination and simulation code."""

from copy import copy
import numpy as np
from numpy.linalg import norm
import pyvista as pv

from raven.planet import Earth
from raven.reference_frame import Frame
from raven.motion import rk4

# TODO: simulate rotation with some dummy geometry (cylindrical with some orientation)
# TODO: consider storing v, omega within a reference frame itself
# TODO: move planet through time as well
# TODO: ^ initialise vehicle velocity relative to planet velocity

def progress_bar(size:int, progress:float) -> str:
    """Draws a progress bar showing percentage complete with a filling bar."""

    p = round(progress*size)
    return f"{progress*100:03.1f}" + '% [' + '/'*p + (size - p)*'-' + ']'

def orbit_sim_rk4(tf:float, dt:float) -> None:
    """Simulate trajectory around a planet using RK4 solver."""

    earth = Earth(x0=np.zeros(3, float), az0=0.0)

    t_arr = np.arange(0.0, tf, dt)
    x_arr = np.zeros((t_arr.size, 3), float)
    v_arr = np.zeros((t_arr.size, 3), float)
    x_arr[0, :] = [earth.r + 100e3, 0, 0]
    v_arr[0, :] = [0, 6000, 6000]

    frame = Frame()
    frame.place_axis_angle(axis=np.array([1, 0, 0], float),
                           ang=0.0,
                           origin=np.array([earth.r + 100e3, 0, 0], float))

    rocket = {'frame':frame, 'v':copy(v_arr[0,:]), 'm':1000.0}

    for i, t in enumerate(t_arr):

        print(progress_bar(20, t / tf) + " (orbit)", end='\r', flush=True)
        x_r = np.r_[rocket['frame'].origin, rocket['v']]
        x_p = np.r_[earth.pci.origin, earth.v]
        x_r = rk4(x_r, x_p, dt, rocket['m'], earth.mu, earth.r)

        rocket['frame'].origin=x_r[:3]
        rocket['v'] = x_r[3:6]

        x_arr[i, :] = x_r[:3]
        v_arr[i, :] = x_r[3:6]

    print("")

    # _, ax = plt.subplots()
    # ax.plot(x_arr[:, 0], x_arr[:, 1], color='r', label='trajectory_xy')
    # ax.add_patch(plt.Circle((0,0), earth.r, color='blue'))
    # ax.set_xlabel("x, m")
    # ax.set_ylabel("y, m")

    # ax.set_aspect('equal')
    # plt.tight_layout()
    # plt.show()

    x_line = pv.MultipleLines(x_arr)
    x_line['v_mag'] = norm(v_arr, axis=1)

    pv.set_plot_theme('dark')
    pl = pv.Plotter()
    pl.set_background('black')
    pl.add_mesh(pv.Sphere(radius=earth.r), color='blue', show_edges=False)
    pl.add_mesh(x_line.tube(radius=50e3), scalars='v_mag', cmap='viridis')
    pl.show()



orbit_sim_rk4(40e3, 0.5)
