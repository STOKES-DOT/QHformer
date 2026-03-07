"""
QHformer Models

This package contains the QHformer model with Inner Product Attention.
"""

from .inner_product_attention import (
    InnerProductAttentionLayer,
    InnerProductAttentionNetLayer,
    InnerProduct,
    NormGate,
    ExponentialBernsteinRBF,
    get_feasible_irrep,
    scatter,
)

from .qhformer import (
    QHformer,
    AttentionQHNet,  # Alias for backward compatibility
    SelfNetLayer,
    PairNetLayer,
)

__all__ = [
    # Main model
    'QHformer',
    'AttentionQHNet',

    # Attention layer
    'InnerProductAttentionLayer',
    'InnerProductAttentionNetLayer',

    # QHNet components
    'SelfNetLayer',
    'PairNetLayer',

    # Utility modules
    'InnerProduct',
    'NormGate',
    'ExponentialBernsteinRBF',
    'get_feasible_irrep',
    'scatter',
]
