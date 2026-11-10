"""Main script for running simulations."""

from os import getcwd
from os.path import join

from raven.simulate import simulate
from raven.plotter import plot_trajectories

def main():
    """Runs a simulation and plots results."""

    results = simulate(join(getcwd(), "input.yaml"))

    if results is None:
        print("Simulation failed.")
        return

    print("Simulation complete, plotting...")
    plot_trajectories(results)


main()
