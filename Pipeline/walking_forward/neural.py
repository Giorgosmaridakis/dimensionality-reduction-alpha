"""Facade mixin combining neural walk-forward helpers."""

from .neural_artifacts import NeuralArtifactsMixin
from .neural_tuning import NeuralTuningMixin
from .neural_utils import NeuralUtilsMixin


class NeuralMixin(NeuralTuningMixin, NeuralArtifactsMixin, NeuralUtilsMixin):
    pass
