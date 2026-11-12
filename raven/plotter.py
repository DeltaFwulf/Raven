"""Various plotting functions."""

from copy import copy
import pyvista as pv
import numpy as np

from .vectors import transform, ntransform, quaternion2rotation_matrix, hamilton, relative_quaternion


# FIXME: calculate the best camera position based on root body
def animate(results:dict, freq:int=30, show_omega:bool=False) -> None:
    """Animates results from a given frame of reference.
    
    freq specifies the frames per second"""

    pv.set_plot_theme('dark')
    pl = pv.Plotter()

    # get root frame x, q arrays
    root = results['runtime']['root_frame'].split('/')
    nsteps = len(results['runtime']['t'])

    if root[0] == 'universal':
        x_root = np.zeros((len(results['runtime']['t']), 3), float)
        q_root = np.tile(np.array([1, 0, 0, 0], float), (nsteps, 1))

    else:
        b_root = results['bodies'][root[0]]
        x_root = b_root['x']

        if root[1] == 'fixed':
            q_root = b_root['q']
        elif b_root['type'] == 0:
            q_root = np.tile(b_root['q_pci'], (nsteps, 1))
        else:
            q_root = np.tile(np.array([1, 0, 0, 0], float), (nsteps, 1))

    actors = {}
    omegas = {}
    for k, b in results['bodies'].items():
        # create actors to manipulate
        if b['type'] == 0:
            # create a sphere of correct radius, colour
            mesh = pv.Sphere(radius=b['r'])
            colour = b['colour']
        else:
            mesh = pv.Cube(x_length=3.0, y_length=2.0, z_length=1.0)
            colour = 'grey'

        if show_omega:
            omega_arrow = pv.Arrow()
            omegas.update({k:pl.add_mesh(omega_arrow)})

        actors.update({k:pl.add_mesh(mesh, color=colour)})

    affine_matrix = np.identity(4, float)

    def callback(step) -> None:
        # update mesh positions to new step, give max for those that are no longer simulated
        step = step % nsteps

        for k, b in results['bodies'].items():

            # put x, q into correct reference frame
            x = transform(b['x'][step, :], q_root[step, :], x_root[step, :], invert=True)
            q = hamilton(q_root[step, :]*np.array([1, -1, -1, -1], float), b['q'][step, :])

            pl.camera.clipping_range = (1, 1e12)
            # set new user_matrix
            affine_matrix[:3, :3] = quaternion2rotation_matrix(q)
            affine_matrix[:3, 3] = x
            actors[k].user_matrix=copy(affine_matrix)

            if show_omega:
                q_omega = hamilton(q, relative_quaternion(np.array([1, 0, 0], float), b['omega'][step,:]))
                affine_matrix[:3, :3] = quaternion2rotation_matrix(q_omega)
                omegas[k].user_matrix = copy(affine_matrix)
                omegas[k].scale = 3*np.linalg.norm(b['omega'][step,:])

    pl.add_timer_event(max_steps=1e6, duration=round(1000 / freq), callback=callback)
    # pl.camera_position = pv.CameraPosition(position=(10, 10, 10), focal_point=(0.0, 0.0, 0.0), viewup=(0.23, 0.072, 0.97),)
    pl.show()


# TODO: use a slider to scrub through time, plot all bodies at their current positions at ends of lines
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
    pl.add_axes(interactive=True)

    cpos = pv.CameraPosition(position=(-38170000, 4602000, 21650000),
                             focal_point=(0.0, 0.0, 0.0),
                             viewup=(0.23, 0.072, 0.97),)

    for b in bodies.values():

        if b is root:
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
