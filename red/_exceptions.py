"""Exceptions for the red package."""


class AnalysisError(Exception):
    """Raised when an error occurs during analysis."""

    ...


class InvalidInputError(Exception):
    """Raised when invalid input data is provided."""

    ...


class EquilibrationNotDetectedError(Exception):
    """Raised when equilibration is not detected."""

    ...
