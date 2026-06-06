"""Structural validation utilities for generated neuron graphs.

Geometry-only port of the validation stack from the sibling `dendrite_gen` repo:
distribution-level Wasserstein-1 metrics (no TMD / tree-edit distance) and
multi-azimuth plot grids. Used by `semlaflow/sample_neurons.py` to compare
generated neuron graphs against the ground-truth split.
"""
