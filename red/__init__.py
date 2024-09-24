"""Robust Equilibration Detection"""

from ._version import __version__
from .confidence_intervals import get_conf_int_init_seq
from .equilibration import (
    detect_equilibration_init_seq,
    detect_equilibration_paired_t_test,
    detect_equilibration_window,
)
from .variance import (
    get_variance_initial_sequence,
    get_variance_series_initial_sequence,
    get_variance_series_window,
    get_variance_window,
)
