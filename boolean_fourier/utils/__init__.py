"""
Shared utilities for Boolean Fourier paper v2.
"""

from .diagnostics import (
    # Support and Jaccard
    support_topk,
    support_nonzero,
    jaccard,
    jaccard_trajectory,
    jaccard_final,

    # Eigenspectrum
    eigenspectrum_svd,
    eigenspectrum_covariance,
    spectral_compression_summary,

    # Routing
    routing_stats,

    # Logging
    DiagnosticsLogger,

    # Protocol
    GD_PROTOCOL,
    get_gd_protocol,

    # Mask loading
    load_phase1_masks,
    load_phase3_masks,
    load_phase4_masks,
)

__all__ = [
    'support_topk',
    'support_nonzero',
    'jaccard',
    'jaccard_trajectory',
    'jaccard_final',
    'eigenspectrum_svd',
    'eigenspectrum_covariance',
    'spectral_compression_summary',
    'routing_stats',
    'DiagnosticsLogger',
    'GD_PROTOCOL',
    'get_gd_protocol',
    'load_phase1_masks',
    'load_phase3_masks',
    'load_phase4_masks',
]
