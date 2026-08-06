"""Shared pytest configuration for the test suite."""

import matplotlib

# Force the non-interactive Agg backend so tests never touch a GUI toolkit
# (Tk/Tcl). This avoids flaky failures on the Windows CI runner, whose Tcl/Tk
# install is broken, and is harmless everywhere else since the plotting tests
# only save figures to disk.
matplotlib.use("Agg")
