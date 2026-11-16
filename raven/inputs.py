"""Functions for initialising simulation data."""

from os.path import join
import numpy as np
import yaml

from .orbit import elements2state
from .vectors import coords2cartesian, cross, transform, hamilton
from . import atmosphere

def parse_inputs(input_path:str) -> tuple[dict, dict] | None:
    """Parses an input file and initialises simulation bodies."""

    with open(input_path, 'r', encoding='utf-8') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    runtime = data.get('runtime')

    root_frame = runtime['root_frame'].split('/')
    if root_frame[0] != 'universal' and root_frame[0] not in data['bodies'].keys():
        print(f"Could not set {runtime['root_frame']} as root frame.")
        return None

    b_def = data.get('bodies') # should preserve order, as newer than python 3.7
    with open(join('raven', 'planets.yaml'), 'r', encoding='utf-8') as f:
        planets = yaml.load(f, Loader=yaml.SafeLoader)

    bodies = {}
    while len(b_def) > 0: # initialise a body, store in bodies, then delete its term in b_def
        loaded = []
        for k, b in b_def.items():

            parent = b.get('parent').split('/')
            if parent[0].lower() != 'universal' and parent[0].lower() not in b_def.keys():
                print(f"Body {k} is orphaned (no body named {parent[0]}).")
                return None

            # skip bodies whose parents have not yet been initialised
            if parent[0].lower() != 'universal' and bodies.get(parent[0]) is None:
                continue

            x_parent = np.zeros(3, float)
            v_parent = np.zeros(3, float)
            q_parent = np.array([1.0, 0.0, 0.0, 0.0], float)

            if parent[0].lower() != 'universal':
                p_body = bodies[parent[0]]
                x_parent = p_body['state'][0:3]
                v_parent = p_body['state'][3:6]

                if parent[1] == 'fixed' and not b.get('orbit') or b.get('coords'):
                    q_parent = p_body['state'][6:10]
                else:
                    q_parent = p_body['q_pci']

            body = {}

            xb = np.zeros(3, float)
            vb = np.zeros(3, float)

            state = np.zeros(14, float)
            if b.get('rv'):
                xb = b['rv'][0:3]
                vb = b['rv'][3:6]

            elif b.get('orbit'):
                xb, vb = elements2state(b['orbit'], p_body['m']*6.67408e-11)

            elif b.get('coords'): # body is fixed w.r.t. the fixed frame
                xb = coords2cartesian(np.rad2deg(b['coords'][0]),
                                                 np.rad2deg(b['coords'][1]),
                                                 p_body['r'] + b['coords'][2])

                # get local components from NEU vector
                x_unit = xb / np.linalg.norm(xb)
                e = cross(np.array([0, 0, 1]), x_unit)
                n = cross(x_unit, e)
                vb = cross(p_body['state'][10:13], xb) + n*b['coords'][3] + e*b['coords'][4] + x_unit*b['coords'][5]

            state[0:3] = transform(xb, q_parent, x_parent)
            state[3:6] = transform(vb, q_parent, v_parent)

            if b['type'] == 0:
                b.update(planets[b['planet']])
                body.update({'r':b['r'], 'colour':b['colour'], 'm':b['m']})
                body.update({'q_pci':hamilton(q_parent, np.array(b['q_pci']))})
                state[6:10] = hamilton(body['q_pci'], np.array(b['q']))

            else:
                state[6:10] = hamilton(q_parent, np.array(b['q']))

            state[10:13] = b['omega']
            state[13] = b['m']
            body.update({'state':state})

            # properties dict
            props = {'type':b['type']}
            if props['type'] == 0:
                atmo_model = getattr(atmosphere, b['atmosphere'], None)
                props.update({'r':b['r'], 'atmosphere':atmo_model})
            elif props['type'] == 1:
                props.update({'cda':b.get('cda')}) # NOTE: this is sometimes None

            body.update({'properties':props})

            # perturbation array
            body.update({'perturbations':np.r_[np.zeros(6), b['I'], 0]})
            bodies.update({k:body})
            loaded.append(k)

        for l in loaded:
            b_def.pop(l)

    return runtime, bodies
