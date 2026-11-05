"""Various plotting functions."""

import pyvista as pv

from .vectors import rotate_quaternion

def animate(results:dict) -> None:
    """Plots the system through time."""

    time = results['t']
    bodies = results['bodies']

    actors = {}
    pl = pv.Plotter()

    for key, body in bodies.items():
        # make it a sphere of correct radius if planet
        if body['type'] == 0:
            z = rotate_quaternion(np.array([0, 0, 1], float), body['state'][0, 6:10])
            sphere = pv.Sphere(radius=body['radius'], centre=body['state'][0, :3], direction=z)
            actors.update({key:pl.add_mesh(sphere)})

        elif body['type'] == 1:
            centre = 
            cyl = pv.Cylinder(c)
        # make it a cylinder big enough to see if rocket

    def callback(step):
