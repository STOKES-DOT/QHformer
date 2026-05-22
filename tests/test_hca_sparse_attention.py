import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from e3nn import o3


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.inner_product_attention import HeavyCompressedAttentionLayer  # noqa: E402


def test_hca_projects_key_value_only_on_topk_edges():
    layer = HeavyCompressedAttentionLayer(
        irrep_in_node=o3.Irreps("4x0e + 4x1o"),
        irrep_hidden=o3.Irreps("4x0e + 4x1o"),
        irrep_out=o3.Irreps("4x0e + 4x1o"),
        sh_irrep=o3.Irreps("0e"),
        edge_attr_dim=4,
        node_attr_dim=4,
        invariant_layers=1,
        invariant_neurons=8,
        use_norm_gate=False,
        num_heads=2,
        hca_lmax=1,
        top_k=1,
        indexer_compress_dim=4,
    )
    layer.eval()

    data = SimpleNamespace(
        edge_index=torch.tensor(
            [
                [0, 0, 1, 1, 2, 2],
                [1, 2, 0, 2, 0, 1],
            ],
            dtype=torch.long,
        ),
        edge_sh=torch.ones(6, 1),
        edge_attr=torch.randn(6, 4),
    )
    x = torch.randn(3, layer.irrep_in_node.dim)
    captured = {}

    def fake_project_key_value(x_arg, edge_src, edge_sh, edge_attr, s0):
        captured["num_projected_edges"] = int(edge_src.numel())
        captured["edge_attr_rows"] = int(edge_attr.shape[0])
        return (
            torch.zeros(edge_src.numel(), layer.irrep_tp_key.dim),
            torch.zeros(edge_src.numel(), layer.irrep_tp_value.dim),
        )

    def fake_attention(query, key, value, edge_dst, key_irrep, value_irrep, num_nodes):
        captured["num_attention_edges"] = int(edge_dst.numel())
        return torch.zeros(num_nodes, layer.irrep_tp_value.dim)

    layer._project_key_value = fake_project_key_value
    layer._multihead_attention = fake_attention

    with torch.no_grad():
        layer(data, x)

    assert captured["num_projected_edges"] == 3
    assert captured["edge_attr_rows"] == 3
    assert captured["num_attention_edges"] == 3
