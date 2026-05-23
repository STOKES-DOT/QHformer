import torch
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.torch_ops import radius_graph, scatter


def test_radius_graph_keeps_directed_edges_inside_each_batch():
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    batch = torch.tensor([0, 0, 0, 1])

    edge_index = radius_graph(pos, r=1.1, batch=batch, loop=False)

    edges = {tuple(edge.tolist()) for edge in edge_index.t()}
    assert edges == {(0, 1), (1, 0)}


def test_scatter_matches_core_sum_mean_max_semantics():
    src = torch.tensor([[1.0, 4.0], [3.0, 2.0], [-1.0, 5.0]])
    index = torch.tensor([0, 0, 1])

    assert torch.allclose(
        scatter(src, index, dim=0, dim_size=3, reduce="sum"),
        torch.tensor([[4.0, 6.0], [-1.0, 5.0], [0.0, 0.0]]),
    )
    assert torch.allclose(
        scatter(src, index, dim=0, dim_size=3, reduce="mean"),
        torch.tensor([[2.0, 3.0], [-1.0, 5.0], [0.0, 0.0]]),
    )
    assert torch.allclose(
        scatter(src, index, dim=0, dim_size=3, reduce="max"),
        torch.tensor([[3.0, 4.0], [-1.0, 5.0], [0.0, 0.0]]),
    )
