"""Main script for running simulations."""

from os import getcwd
from os.path import join

from raven.simulate import simulate
from raven.plotter import plot_trajectories #animate

def main():
    """Runs a simulation and plots results."""

    results = simulate(join(getcwd(), "inputs", "free_return.yaml"))

    if results is None:
        print("Simulation failed.")
        return

    print("Simulation complete, plotting...")
    plot_trajectories(results)
    # animate(results, freq=30, show_omega=True)


main()
