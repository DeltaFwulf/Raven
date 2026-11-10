"""Various plotting functions."""

import pyvista as pv
import numpy as np

from .vectors import ntransform, cross


def plot_trajectories(results:dict) -> None:
    """Plots all trajectories w.r.t. the root frame."""

    runtime = results['runtime']
    bodies = results['bodies']

    root_frame = runtime['root_frame'].split('/')
    root_origin = root_frame[0]
    root_orientation = root_frame[1]
    root = bodies.get(root_origin)

    # pos = np.array([1e7, 1e7, 1e7])
    # x_cam = -pos / np.linalg.norm(pos)
    # y_cam = cross(np.array([0,0,1]), x_cam)
    # z_cam = cross(x_cam, y_cam / np.linalg.norm(y_cam))

    pv.set_plot_theme('dark')
    pl = pv.Plotter()
    pl.add_axes(interactive=True)

    cpos = pv.CameraPosition(position=(-38170000, 4602000, 21650000),
                             focal_point=(0.0, 0.0, 0.0),
                             viewup=(0.23, 0.072, 0.97),)

    for b in bodies.values():

        if b is root:
            # TODO: get a shape def for non-planets
            shape = pv.Sphere(radius=b['r'])
            pl.add_mesh(shape, color=root['colour'])
            continue

        if root is not None:
            q_root = root['q'] if root_orientation == 'fixed' else np.tile(root['q_pci'], (np.shape(root['q'])[0], 1))
            pts = ntransform(b['x'], q_root, root['x'], invert=True)
            v = ntransform(b['v'], q_root, root['v'], invert=True)
        else:
            pts = b['x']
            v = b['v']

        line = pv.MultipleLines(points=pts)
        line['v_mag'] = np.linalg.norm(v, axis=1)
        pl.add_mesh(line.tube(radius=5e4), scalars='v_mag', cmap='viridis')

    pl.camera_position = cpos
    pl.show()
