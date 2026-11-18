"""Various plotting functions."""

from copy import copy
import pyvista as pv
import numpy as np

from .vectors import ntransform, quaternion2rotation_matrix, hamilton, relative_quaternion


# FIXME: calculate the best camera position based on root body
def animate(results:dict, **kwargs) -> None:
    """Animates results from a given frame of reference.
    
    freq specifies the frames per second"""

    pv.set_plot_theme('dark')
    pl = pv.Plotter()

    freq = 30 if not kwargs.get('freq') else kwargs['freq']
    show_omega = False if not kwargs.get('show_omega') else kwargs['show_omega']

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
        if b['properties']['type'] == 0:
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
            x = ntransform(b['x'][step, :], q_root[step, :], x_root[step, :], invert=True)
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


# TODO: draw rockets as a blown up version of its defined shape
# TODO: add a slider for line tube thickness
# TODO: time slider should show time bounds rather than progress
def scrub_trajectory(results:dict, **kwargs) -> None:
    """Plots the trajectory up to the time defined by a user, via a slider widget."""

    runtime = results['runtime']
    bodies = results['bodies']

    root_frame = runtime['root_frame'].split('/')
    root_body = bodies.get(root_frame[0])
    
    if root_body is None and runtime['root_frame'] != 'universal':
        raise ValueError("Invalid root frame given.")

    root_orientation = root_frame[-1]
    pv.set_plot_theme('dark')
    pl = pv.Plotter()

    def scrub(progress:float) -> None:
        """Scrubs through results timeline, plots state at given fraction of total time."""

        n = max(2, round(len(runtime['t'])*progress / 100.0))

        for k, b in bodies.items():

            # Draw trajectories of all non-root bodies up to current step
            if root_body is not None:
                q_root = root_body['q'][:n, :] if root_orientation == 'fixed' else np.tile(root_body['q_pci'], (n, 1))
                pts = ntransform(b['x'][:n, :], q_root, root_body['x'][:n, :], invert=True)
                v = ntransform(b['v'][:n, :], q_root, root_body['v'][:n, :], invert=True)
            else:
                pts = b['x'][:n, :]
                v = b['v'][:n, :]

            # Give bodies a shape
            if b['properties']['type'] == 0:
                shape = pv.Sphere(radius=b['r'], center=pts[-1, :])
                pl.add_mesh(shape, color=b['colour'], name=k + "_shape")

            if b is root_body:
                continue

            line = pv.MultipleLines(points=pts)
            line['v_mag'] = np.linalg.norm(v, axis=1)
            pl.add_mesh(line.tube(radius=2e4), scalars='v_mag', cmap='viridis', name=k + "_line")


    pl.add_slider_widget(scrub, [0, 100], title="Progress (%)")
    pl.show()
