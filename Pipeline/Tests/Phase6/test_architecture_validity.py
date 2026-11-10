"""
Phase 6: Architecture Validity
Requirement: Verify that the latent layer (the bottleneck) has exactly the dimensions specified in the latent_dim config.
"""

from __future__ import annotations
import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase6.test_config import ae_vanilla_context

def test_architecture_validity_dimensions_ae(ae_vanilla_context):
    """Verifies K conditional beta components for GKX conditional AE."""
    ctx = ae_vanilla_context
    df = ctx["df_transformed"]
    k = ctx["latent_dim"]
    cols = [c for c in df.columns if c.startswith('ae_component_')]
    assert len(cols) == k, f"AE expected {k} components, got {len(cols)}"

def test_architecture_internal_layers_ae(ae_vanilla_context):
    """Checks the internal PyTorch layer dimensions for GKX conditional AE."""
    model = ae_vanilla_context["model"]
    k = ae_vanilla_context["latent_dim"]
    # In GKXConditionalAutoencoder, the final beta_net layer outputs K betas.
    last_layer = model.beta_net[-1]
    assert last_layer.out_features == k
