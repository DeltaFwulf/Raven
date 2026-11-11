"""Code performance profiling."""

import cProfile
import pstats
from pstats import SortKey

from os import getcwd
from os.path import join

from raven.simulate import simulate


def raven_noplot():
    """Runs a simulation and plots results."""

    results = simulate(join(getcwd(), "input.yaml"))

    if results is None:
        print("Simulation failed.")
        return

cProfile.run('raven_noplot()', 'stats')
p = pstats.Stats('stats')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
