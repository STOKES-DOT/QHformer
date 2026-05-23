"""
QHformer Models

This package contains the QHformer model with Inner Product Attention.
"""

from .inner_product_attention import (
    MultiHeadAttentionLayer,
    MultiHeadAttentionNetLayer,
    CompressedSparseAttentionLayer,
    CompressedSparseAttentionNetLayer,
    HeavyCompressedAttentionLayer,
    HeavyCompressedAttentionNetLayer,
    MultiHeadInnerProduct,
    MultiHeadEquivariantNorm,
    InvariantAttentionScore,
    InnerProduct,
    NormGate,
    ExponentialBernsteinRBF,
    get_feasible_irrep,
    split_irreps_multiplicity,
    merge_heads,
    scatter,
)

from .qhformer import (
    QHformer,
    AttentionQHNet,  # Alias for backward compatibility
    SelfNetLayer,
    PairNetLayer,
)
from .so2_ops import SO2EdgeConv, init_edge_frame

__all__ = [
    # Main model
    'QHformer',
    'AttentionQHNet',

    # Attention layers
    'MultiHeadAttentionLayer',
    'MultiHeadAttentionNetLayer',
    'CompressedSparseAttentionLayer',
    'CompressedSparseAttentionNetLayer',
    'HeavyCompressedAttentionLayer',
    'HeavyCompressedAttentionNetLayer',
    'MultiHeadInnerProduct',
    'MultiHeadEquivariantNorm',
    'InvariantAttentionScore',

    # QHNet components
    'SelfNetLayer',
    'PairNetLayer',
    'SO2EdgeConv',
    'init_edge_frame',

    # Utility modules
    'InnerProduct',
    'NormGate',
    'ExponentialBernsteinRBF',
    'get_feasible_irrep',
    'split_irreps_multiplicity',
    'merge_heads',
    'scatter',
]
