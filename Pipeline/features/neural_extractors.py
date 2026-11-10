"""
Neural feature extraction facade used by the walk-forward forecaster.

The GKX and FactorVAE implementations live in sibling modules. This facade
keeps the public fit_transform/transform API stable for existing callers.
"""

try:
    from .factor_vae_extractor import FactorVAEExtractorMixin
    from .gkx_extractor import GKXAutoencoderMixin
except ImportError:  # Supports running Pipeline modules as scripts.
    from features.factor_vae_extractor import FactorVAEExtractorMixin
    from features.gkx_extractor import GKXAutoencoderMixin


class NeuralFeatureExtractor(GKXAutoencoderMixin, FactorVAEExtractorMixin):
    def __init__(self):
        # These fitted model objects are set by fit_transform_* and reused by
        # transform_* on validation/test folds.
        self.autoencoder = None
        self.factor_vae_model = None
