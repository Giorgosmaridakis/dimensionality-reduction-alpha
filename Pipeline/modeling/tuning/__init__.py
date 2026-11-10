"""Nested hyperparameter tuning package.

The public import remains:

    from modeling.tuning import HyperparameterTuner
"""

# Re-export the public class from the package root. This keeps old imports from
# the former single-file module working:
#   from modeling.tuning import HyperparameterTuner
#
# Model is also re-exported because older tests/scripts patch
# modeling.tuning.Model when they want to replace temporary fold models.
from .tuner import HyperparameterTuner, Model

__all__ = ["HyperparameterTuner", "Model"]
