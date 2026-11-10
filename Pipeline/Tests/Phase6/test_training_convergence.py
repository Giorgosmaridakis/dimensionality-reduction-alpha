"""
Phase 6: Training Convergence
Requirement: Verify that the Loss Function (usually MSE) is tracked and reaches a stable floor after several epochs.
"""

from __future__ import annotations
import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))
from Tests.Phase6.test_config import ae_vanilla_context

def verify_convergence(ctx):
    """Internal helper to verify loss decreases over epochs."""
    hist = ctx["history"]
    assert "train_loss" in hist, "Training loss not recorded in history."
    losses = hist["train_loss"]
    assert len(losses) >= 2, "Need at least 2 epochs to verify trend."
    # Ensure final loss is not significantly worse than start
    assert losses[-1] <= losses[0] * 1.05, f"Loss spiked at end: {losses[0]} -> {losses[-1]}"

def test_training_convergence_ae(ae_vanilla_context):
    """Verifies convergence for Vanilla Autoencoder."""
    verify_convergence(ae_vanilla_context)
