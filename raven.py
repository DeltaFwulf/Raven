"""Main script for running simulations."""

from os import getcwd
from os.path import join

from raven.simulate import simulate
from raven.plotter import animate

def main():
    """Runs a simulation and plots results."""

    results = simulate(join(getcwd(), "inputs", "rotation_test.yaml"))

    if results is None:
        print("Simulation failed.")
        return

    print("Simulation complete, plotting...")
    #plot_trajectories(results)
    animate(results, freq=30, show_omega=True)


main()
