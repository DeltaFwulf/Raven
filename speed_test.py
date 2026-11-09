"""Code performance profiling."""

import profiling.tracing
import pstats
from pstats import SortKey

#pylint: disable=W0611
from raven.nbody import simulate

profiling.tracing.run('simulate("/home/deltav/Documents/GitHub/Raven/input.yaml")', 'stats')
p = pstats.Stats('stats')
p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)
