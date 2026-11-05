"""Orbit determination and simulation code."""

from os import getcwd
from os.path import join
import numpy as np
from numpy.linalg import norm
import pyvista as pv
import yaml

from raven.motion import rk4
from raven.vectors import hamilton


def progress_bar(size:int, progress:float) -> str:
    """Draws a progress bar showing percentage complete with a filling bar."""
    p = round(progress*size)
    return f"{progress*100:03.1f}" + '% [' + '/'*p + (size - p)*'-' + ']'


def load_inputs(input_path:str) -> tuple[dict, dict]:
    """Parses an input file and initialises simulation bodies."""

    with open(input_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    runtime = data.get('runtime')
    runtime.update({'t':np.arange(0.0, runtime['tf'], runtime['dt'])})

    bodies = data.get('bodies') # should preserve order, as newer than python 3.7
    with open(join('raven', 'planets.yaml'), 'r', encoding='utf-8') as f:
        planets = yaml.load(f, Loader=yaml.SafeLoader)

    for b in bodies.values():

        if b.get('type') == 0:
            b.update(planets.get(b.get('planet')))
            q_az = np.r_[np.cos(b.get('az') / 2), np.sin(b.get('az') / 2)*np.array([0, 0, 1], float)]
            b.update({'q':hamilton(np.array(b.get('q_pci')), q_az)})

        # pack state array (process frame translation here too)
        state = np.r_[b.get('x'),
                      b.get('v'),
                      b.get('q'),
                      b.get('omega'),
                      b.get('m')]

        b.update({'state':state})

        internals = np.r_[np.zeros(6), b['I'], 0, b['type']]
        b.update({'internals':internals})

    return runtime, bodies


def simulate(input_path:str) -> None:
    """Simulate trajectory around a planet using RK4 solver."""

    runtime, bodies = load_inputs(input_path)
    state_offset = 14
    internals_offset = 11

    # initialise system state array
    system_state = np.zeros((len(bodies)*state_offset), float)
    internals = np.zeros((len(bodies)*internals_offset), float)
    for n,b in enumerate(bodies.values()):
        system_state[n*state_offset:(n + 1)*state_offset] = b.get('state')
        internals[n*internals_offset:(n + 1)*internals_offset] = b.get('internals')

    t_saved = 0.0
    for i, t in enumerate(runtime['t']):

        print(progress_bar(20, t / runtime['tf']) + " (orbit)", end='\r', flush=True)

        # update system state
        system_state = rk4(system_state, internals, runtime['dt'])

        if np.isnan(system_state).any():
            print("state contains NaN")
            break

        # update body state arrays
        save = False
        if t - t_saved >= runtime['dt_store']:
            save = True
            t_saved = t

        for n, b in enumerate(bodies.values()):

            body_state = system_state[n*state_offset:(n + 1)*state_offset]
            if save:
                b['state'] = np.vstack((b['state'], body_state))

            # calculate body internals

            # pack internals array

    # Convert to desired reference frame
    # root_path = bodies.get(runtime.get('root'))['state']

    # plot all paths relative to root frame

    pv.set_plot_theme('dark')
    pl = pv.Plotter()
    pl.set_background('black')
    # pl.add_mesh(pv.Sphere(radius=bodies['earth']['r']), color='blue', show_edges=False)
    for b in bodies.values():
        line = pv.MultipleLines(b['state'][:, :3])
        line['v_mag'] = norm(b['state'][:, 3:6], axis=1)
        pl.add_mesh(line.tube(radius=100e3), scalars='v_mag', cmap='viridis')

    print("")
    pl.show()


simulate(join(getcwd(), "input.yaml"))
