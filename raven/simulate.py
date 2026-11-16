"""Orbit determination and simulation code."""

import numpy as np

from .motion import rk45

def progress_bar(size:int, progress:float) -> str:
    """Draws a progress bar showing percentage complete with a filling bar."""
    p = round(progress*size)
    return f"{progress*100:03.1f}" + '% [' + '/'*p + (size - p)*'-' + ']'


def simulate(runtime:dict, bodies:dict) -> dict | None:
    """Simulate trajectory around a planet using RK4 solver."""

    state_offset = 14
    perturbs_offset = 10

    # initialise system state array
    system_state = np.zeros((len(bodies)*state_offset), float)
    perturbs = np.zeros((len(bodies)*perturbs_offset), float)
    properties = []
    for n,b in enumerate(bodies.values()):
        system_state[n*state_offset:(n + 1)*state_offset] = b.get('state')
        perturbs[n*perturbs_offset:(n + 1)*perturbs_offset] = b.get('perturbations')
        properties.append(b.get('properties'))

    h = runtime['dt']
    h_max = np.inf if not runtime.get('hmax') else runtime['hmax']
    t = [0.0,]

    while t[-1] < runtime['tf']:
        print(progress_bar(20, t[-1] / runtime['tf']) + f" (h = {h:0.1f} s)     ", end='\r', flush=True)

        system_state, h, h_new = rk45(system_state,
                                      perturbs,
                                      properties,
                                      h,
                                      runtime['atol'],
                                      runtime['rtol'],
                                      min(h_max, runtime['tf'] - t[-1]))

        t.append(t[-1] + h)
        h = h_new

        if np.isnan(system_state).any():
            print("state contains NaN")
            return None

        for n, b in enumerate(bodies.values()):
            body_state = system_state[n*state_offset:(n + 1)*state_offset]
            b['state'] = np.vstack((b['state'], body_state))

    print("")

    for body in bodies.values():
        body['x'] = body['state'][:, :3]
        body['v'] = body['state'][:, 3:6]
        body['q'] = body['state'][:, 6:10]
        body['omega'] = body['state'][:, 10:13]
        body['m'] = body['state'][:, 13]

        body.pop("state")
        if body['properties'].get('atmosphere'):
            body['properties'].pop("atmosphere")

    runtime.update({'t':t})
    return {'runtime':runtime, 'bodies':bodies}
