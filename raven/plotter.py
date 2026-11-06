"""Various plotting functions."""

import pyvista as pv
import numpy as np

from .reference_frame import ntransform

# def animate(results:dict) -> None:
#     """Plots the system through time."""

#     time = results['t']
#     bodies = results['bodies']

#     actors = {}
#     pl = pv.Plotter()

#     for key, body in bodies.items():
#         # make it a sphere of correct radius if planet
#         if body['type'] == 0:
#             z = rotate_quaternion(np.array([0, 0, 1], float), body['state'][0, 6:10])
#             sphere = pv.Sphere(radius=body['radius'], centre=body['state'][0, :3], direction=z)
#             actors.update({key:pl.add_mesh(sphere)})

#         elif body['type'] == 1:
#             centre = 0
#             cyl = pv.Cylinder(c)
#         # make it a cylinder big enough to see if rocket

#     def callback(step):


def plot_trajectories(results:dict) -> None:
    """Plots all trajectories w.r.t. the root frame."""

    runtime = results['runtime']
    bodies = results['bodies']


    root_frame = runtime['root_frame'].split('/')
    root_origin = root_frame[0]
    root_orientation = root_frame[1]
    root = bodies.get(root_origin)

    pv.set_plot_theme('dark')
    pl = pv.Plotter()

    for b in bodies.values():

        if b is root:
            # TODO: get a shape def for non-planets
            shape = pv.Sphere(radius=b['r'])
            pl.add_mesh(shape, color='blue')
            continue

        if root is not None:
            q_root = root['q'] if root_orientation == 'fixed' else np.tile(root['q_pci'], (np.shape(root['q'])[0], 1))
            pts = ntransform(b['x'], q_root, root['x'], invert=True)
            v = ntransform(b['v'], q_root, root['v'], invert=True)
            # q = nhamilton(b['q'], root['q'], invert=True)
        else:
            pts = b['x']
            v = b['v']
            # q = b['q']

        line = pv.MultipleLines(points=pts)
        line['v_mag'] = np.linalg.norm(v, axis=1)
        pl.add_mesh(line.tube(radius=5e4), scalars='v_mag', cmap='viridis')

    pl.show()
