"""Orbit determination and simulation code."""

from os import getcwd
from os.path import join
import numpy as np
import yaml

from raven.motion import rk45
from raven.vectors import hamilton
from raven.plotter import plot_trajectories


# TODO: re-think dynamic storage, removing for now


def progress_bar(size:int, progress:float) -> str:
    """Draws a progress bar showing percentage complete with a filling bar."""
    p = round(progress*size)
    return f" {progress*100:03.1f}" + '% [' + '/'*p + (size - p)*'-' + ']'


def load_inputs(input_path:str) -> tuple[dict, dict]:
    """Parses an input file and initialises simulation bodies."""

    with open(input_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    runtime = data.get('runtime')
    # runtime.update({'t':np.arange(0.0, runtime['tf'], runtime['dt'])})

    bodies = data.get('bodies') # should preserve order, as newer than python 3.7
    with open(join('raven', 'planets.yaml'), 'r', encoding='utf-8') as f:
        planets = yaml.load(f, Loader=yaml.SafeLoader)

    for b in bodies.values():

        if b.get('type') == 0:
            b.update(planets.get(b.get('planet')))
            q_az = np.r_[np.cos(b.get('az') / 2), np.sin(b.get('az') / 2)*np.array([0, 0, 1], float)]
            b.update({'q_pci':np.array(b['q_pci'])})
            b.update({'q':hamilton(b['q_pci'], q_az)})

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
    perturbs_offset = 11

    # initialise system state array
    system_state = np.zeros((len(bodies)*state_offset), float)
    perturbs = np.zeros((len(bodies)*perturbs_offset), float)
    for n,b in enumerate(bodies.values()):
        system_state[n*state_offset:(n + 1)*state_offset] = b.get('state')
        perturbs[n*perturbs_offset:(n + 1)*perturbs_offset] = b.get('internals')

    h = runtime['dt']
    t = [0.0,]

    while(t[-1] < runtime['tf']):

        print(progress_bar(20, t[-1] / runtime['tf']) + f" (h = {h:0.3f} s)", end='\r', flush=True)

        # update system state
        # system_state = rk4(system_state, perturbs, runtime['dt'])
        system_state, h, h_new = rk45(system_state,
                               perturbs,
                               h,
                               runtime['atol'],
                               runtime['rtol'],
                               runtime['tf'] - t[-1])

        t.append(t[-1] + h)
        h = h_new

        if np.isnan(system_state).any():
            print("state contains NaN")
            return

        # update body state arrays
        for n, b in enumerate(bodies.values()):
            body_state = system_state[n*state_offset:(n + 1)*state_offset]
            b['state'] = np.vstack((b['state'], body_state))

            # update internal loads, etc


    print("")

    for body in bodies.values():
        body['x'] = body['state'][:, :3]
        body['v'] = body['state'][:, 3:6]
        body['q'] = body['state'][:, 6:10]
        body['omega'] = body['state'][:, 10:13]
        body['m'] = body['state'][:, 13]

        body.pop("state")

    runtime.update({'t':t})

    results = {'runtime':runtime, 'bodies':bodies}
    plot_trajectories(results)


simulate(join(getcwd(), "input.yaml"))
