"""Code performance profiling."""

import cProfile
import pstats
from pstats import SortKey

from os import getcwd
from os.path import join
from datetime import datetime, timezone

from raven.simulate import simulate


def raven_noplot():
    """Runs a simulation and plots results."""

    results = simulate(join(getcwd(), "input.yaml"))

    if results is None:
        print("Simulation failed.")
        return

dt = datetime.now(timezone.utc)
FILENAME = f"raven_profile_{dt.strftime("%F_%H%M%S")}.prof"
cProfile.run('raven_noplot()', FILENAME)
p = pstats.Stats(FILENAME)
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
