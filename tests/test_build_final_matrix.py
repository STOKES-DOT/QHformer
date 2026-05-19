import sys
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.qhformer import QHformer  # noqa: E402


class BatchLike:
    pass


def _full_edges(num_nodes, offset):
    pairs = [
        (dst + offset, src + offset)
        for dst in range(num_nodes)
        for src in range(num_nodes)
        if dst != src
    ]
    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _slow_reference(model, data, diagonal_matrix, non_diagonal_matrix):
    final_matrix = []
    dst, src = data.full_edge_index
    for graph_idx in range(data.ptr.shape[0] - 1):
        matrix_block_col = []
        for src_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
            matrix_col = []
            for dst_idx in range(data.ptr[graph_idx], data.ptr[graph_idx + 1]):
                if src_idx == dst_idx:
                    matrix_col.append(diagonal_matrix[src_idx].index_select(
                        -2, model.orbital_mask[data.atoms[dst_idx].item()]).index_select(
                        -1, model.orbital_mask[data.atoms[src_idx].item()])
                    )
                else:
                    mask1 = (src == src_idx)
                    mask2 = (dst == dst_idx)
                    index = torch.where(mask1 & mask2)[0].item()
                    block = non_diagonal_matrix[index]
                    block = block.index_select(-2, model.orbital_mask[data.atoms[dst_idx].item()])
                    block = block.index_select(-1, model.orbital_mask[data.atoms[src_idx].item()])
                    matrix_col.append(block)
            matrix_block_col.append(torch.cat(matrix_col, dim=-2))
        final_matrix.append(torch.cat(matrix_block_col, dim=-1))
    return torch.stack(final_matrix, dim=0)


def test_build_final_matrix_matches_original_block_ordering():
    torch.manual_seed(0)
    model = QHformer(
        in_node_features=1,
        hidden_size=16,
        bottle_hidden_size=4,
        num_gnn_layers=4,
        radius_embed_dim=8,
        num_heads=4,
        hca_lmax=3,
    )
    model.set(torch.device("cpu"))

    data = BatchLike()
    data.ptr = torch.tensor([0, 3, 6], dtype=torch.long)
    data.atoms = torch.tensor([[8], [1], [1], [6], [1], [1]], dtype=torch.long)
    data.full_edge_index = torch.cat([_full_edges(3, 0), _full_edges(3, 3)], dim=1)

    diagonal = torch.randn(6, 14, 14)
    non_diagonal = torch.randn(data.full_edge_index.shape[1], 14, 14)

    expected = _slow_reference(model, data, diagonal, non_diagonal)
    actual = model.build_final_matrix(data, diagonal, non_diagonal)

    assert torch.allclose(actual, expected)
