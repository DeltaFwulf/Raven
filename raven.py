"""Main script for running simulations."""

from os import getcwd
from os.path import join

from raven.inputs import parse_inputs
from raven.simulate import simulate
from raven import plotter



def main(name:str) -> int:
    """Runs a simulation and plots results."""

    try:
        runtime, bodies = parse_inputs(join(getcwd(), "inputs", name + ".yaml"))
    except TypeError:
        print("Inputs could not be parsed.")
        return 1

    results = simulate(runtime, bodies)

    if results is None:
        print("Simulation failed.")
        return 2

    print("Simulation complete, plotting...")
    plot_fn = getattr(plotter, runtime['plotter']['name'], plotter.scrub_trajectory)
    plot_args = {} if not runtime['plotter'].get('args') else runtime['plotter']['args']
    plot_fn(results, **plot_args)
    return 0


main("free_return")
