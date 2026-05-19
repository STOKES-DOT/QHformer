import os
import sys

import torch
from e3nn import o3
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.inner_product_attention import (
    CompressedSparseAttentionLayer,
    HeavyCompressedAttentionLayer,
    InvariantAttentionScore,
    MultiHeadAttentionLayer,
    NormGate,
    merge_heads,
    split_irreps_multiplicity,
)
from models.qhformer import QHformer


def _mock_edge_data(num_nodes=5, edge_attr_dim=8, sh_lmax=4):
    pos = torch.randn(num_nodes, 3)
    dst, src = [], []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dst.append(i)
                src.append(j)
    edge_index = torch.tensor([dst, src], dtype=torch.long)
    edge_vec = pos[edge_index[0]] - pos[edge_index[1]]
    edge_attr = torch.randn(edge_index.shape[1], edge_attr_dim) * 0.01
    edge_attr[:, 0] = edge_vec.norm(dim=-1)
    sh_irrep = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
    edge_sh = o3.spherical_harmonics(
        sh_irrep,
        edge_vec[:, [1, 2, 0]],
        normalize=True,
        normalization="component",
    )
    return Data(edge_index=edge_index, edge_attr=edge_attr, edge_sh=edge_sh)


def test_split_and_merge_irreps_follow_multiplicity_axis():
    irreps = o3.Irreps("8x0e + 8x1o + 8x2e + 8x3o")
    x = torch.randn(3, irreps.dim)

    split = split_irreps_multiplicity(x, irreps, num_heads=4)
    merged = merge_heads(split, irreps, num_heads=4)

    assert split.shape == (3, 4, irreps.dim // 4)
    assert torch.allclose(merged, x)


def test_invariant_attention_score_starts_as_inner_product_sum_and_can_learn():
    torch.manual_seed(3)
    scorer = InvariantAttentionScore(num_invariants=6, num_heads=4, hidden_dim=8)
    attention_input = torch.randn(7, 6, 4)

    initial_logits = scorer(attention_input)
    assert torch.allclose(initial_logits, attention_input.sum(dim=1))

    with torch.no_grad():
        scorer.output.weight.fill_(0.05)

    learned_logits = scorer(attention_input)
    assert learned_logits.shape == (7, 4)
    assert not torch.allclose(learned_logits, attention_input.sum(dim=1))


def test_invariant_attention_score_can_start_with_nonzero_residual():
    torch.manual_seed(4)
    scorer = InvariantAttentionScore(
        num_invariants=6,
        num_heads=4,
        hidden_dim=8,
        residual_init_std=0.02,
    )
    attention_input = torch.randn(7, 6, 4)

    logits = scorer(attention_input)

    assert torch.count_nonzero(scorer.output.weight).item() > 0
    assert logits.shape == (7, 4)
    assert not torch.allclose(logits, attention_input.sum(dim=1))


def test_attention_layers_initialize_learnable_score_as_noop_residual():
    irrep = o3.Irreps("8x0e + 8x1o + 8x2e")
    sh_irrep = o3.Irreps.spherical_harmonics(lmax=2)
    layer = MultiHeadAttentionLayer(
        irrep_in_node=irrep,
        irrep_hidden=irrep,
        irrep_out=irrep,
        sh_irrep=sh_irrep,
        edge_attr_dim=8,
        node_attr_dim=8,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinear="ssp",
        use_norm_gate=True,
        attention_temperature=1.0,
        num_heads=4,
    )

    assert torch.count_nonzero(layer.attention_score.output.weight).item() == 0
    assert torch.count_nonzero(layer.attention_score.output.bias).item() == 0


def test_attention_layers_can_initialize_score_with_nonzero_residual():
    torch.manual_seed(5)
    irrep = o3.Irreps("8x0e + 8x1o + 8x2e")
    sh_irrep = o3.Irreps.spherical_harmonics(lmax=2)
    layer = MultiHeadAttentionLayer(
        irrep_in_node=irrep,
        irrep_hidden=irrep,
        irrep_out=irrep,
        sh_irrep=sh_irrep,
        edge_attr_dim=8,
        node_attr_dim=8,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinear="ssp",
        use_norm_gate=True,
        attention_temperature=1.0,
        num_heads=4,
        attention_score_residual_init_std=0.02,
    )

    assert torch.count_nonzero(layer.attention_score.output.weight).item() > 0


def test_norm_gate_initializes_close_to_identity_for_equivariant_channels():
    torch.manual_seed(6)
    irreps = o3.Irreps("8x0e + 8x1o + 8x2e")
    gate = NormGate(irreps)
    x = torch.randn(5, irreps.dim)

    out = gate(x)

    assert out.shape == x.shape
    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_attention_residual_uses_original_features_when_message_projection_is_zero():
    torch.manual_seed(6)
    irrep = o3.Irreps("8x0e + 8x1o + 8x2e")
    sh_irrep = o3.Irreps.spherical_harmonics(lmax=2)
    data = _mock_edge_data(num_nodes=5, edge_attr_dim=8, sh_lmax=2)
    x = torch.randn(5, irrep.dim) * 0.1
    layer = MultiHeadAttentionLayer(
        irrep_in_node=irrep,
        irrep_hidden=irrep,
        irrep_out=irrep,
        sh_irrep=sh_irrep,
        edge_attr_dim=8,
        node_attr_dim=8,
        invariant_layers=1,
        invariant_neurons=8,
        nonlinear="ssp",
        use_norm_gate=True,
        attention_temperature=1.0,
        num_heads=4,
    )

    with torch.no_grad():
        for param in layer.linear_out.parameters():
            param.zero_()

    out = layer(data, x)

    assert torch.allclose(out, x, atol=1e-6, rtol=1e-6)


def test_hybrid_attention_layers_return_full_finite_irreps():
    torch.manual_seed(7)
    irrep = o3.Irreps("8x0e + 8x1o + 8x2e + 8x3o + 8x4e")
    sh_irrep = o3.Irreps.spherical_harmonics(lmax=4)
    data = _mock_edge_data(num_nodes=5, edge_attr_dim=8, sh_lmax=4)
    x = torch.randn(5, irrep.dim) * 0.1

    layer_classes = [
        MultiHeadAttentionLayer,
        CompressedSparseAttentionLayer,
        HeavyCompressedAttentionLayer,
    ]
    for layer_cls in layer_classes:
        kwargs = {}
        if layer_cls is CompressedSparseAttentionLayer:
            kwargs.update(top_k=2, indexer_compress_dim=8)
        if layer_cls is HeavyCompressedAttentionLayer:
            kwargs.update(hca_lmax=2)

        layer = layer_cls(
            irrep_in_node=irrep,
            irrep_hidden=irrep,
            irrep_out=irrep,
            sh_irrep=sh_irrep,
            edge_attr_dim=8,
            node_attr_dim=8,
            invariant_layers=1,
            invariant_neurons=8,
            nonlinear="ssp",
            use_norm_gate=True,
            attention_temperature=1.0,
            num_heads=4,
            **kwargs,
        )

        out = layer(data, x)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()


def test_qhformer_uses_csa_hca_csa_hca_pattern():
    model = QHformer(
        in_node_features=1,
        sh_lmax=4,
        hidden_size=4,
        bottle_hidden_size=4,
        num_gnn_layers=4,
        max_radius=6,
        radius_embed_dim=8,
        attention_temperature=1.0,
        num_heads=4,
        use_hybrid_attention=True,
        csa_top_k=2,
        hca_lmax=2,
        indexer_compress_dim=8,
    )

    layer_types = [type(layer.conv).__name__ for layer in model.e3_gnn_layer]
    assert layer_types == [
        "CompressedSparseAttentionLayer",
        "HeavyCompressedAttentionLayer",
        "CompressedSparseAttentionLayer",
        "HeavyCompressedAttentionLayer",
    ]


def test_full_qhformer_hamiltonian_rotation_invariants():
    torch.manual_seed(11)
    model = QHformer(
        in_node_features=1,
        sh_lmax=4,
        hidden_size=4,
        bottle_hidden_size=4,
        num_gnn_layers=4,
        max_radius=8,
        radius_embed_dim=8,
        attention_temperature=1.0,
        num_heads=4,
        use_hybrid_attention=True,
        csa_top_k=2,
        hca_lmax=2,
        indexer_compress_dim=8,
    )
    model.eval()
    model.set("cpu")

    atoms = torch.tensor([[8], [1], [1]], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ],
        dtype=torch.float32,
    )
    data = Data(
        pos=pos,
        atoms=atoms,
        batch=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3], dtype=torch.long),
    )

    rotation = o3.rand_matrix().to(pos.dtype)
    data_rot = data.clone()
    data_rot.pos = pos @ rotation.T

    with torch.no_grad():
        h_original = model(data)["hamiltonian"][0]
        h_rotated = model(data_rot)["hamiltonian"][0]

    assert (h_rotated - h_original).abs().max().item() > 0.0
    assert torch.allclose(torch.trace(h_original), torch.trace(h_rotated), atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        torch.linalg.eigvalsh(h_original),
        torch.linalg.eigvalsh(h_rotated),
        atol=1e-5,
        rtol=1e-5,
    )


def test_nonzero_attention_score_init_preserves_hamiltonian_rotation_invariants():
    torch.manual_seed(13)
    model = QHformer(
        in_node_features=1,
        sh_lmax=4,
        hidden_size=4,
        bottle_hidden_size=4,
        num_gnn_layers=4,
        max_radius=8,
        radius_embed_dim=8,
        attention_temperature=1.0,
        num_heads=4,
        use_hybrid_attention=True,
        csa_top_k=2,
        hca_lmax=2,
        indexer_compress_dim=8,
        attention_score_residual_init_std=0.02,
    )
    model.eval()
    model.set("cpu")

    atoms = torch.tensor([[8], [1], [1]], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ],
        dtype=torch.float32,
    )
    data = Data(
        pos=pos,
        atoms=atoms,
        batch=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3], dtype=torch.long),
    )

    rotation = o3.rand_matrix().to(pos.dtype)
    data_rot = data.clone()
    data_rot.pos = pos @ rotation.T

    with torch.no_grad():
        h_original = model(data)["hamiltonian"][0]
        h_rotated = model(data_rot)["hamiltonian"][0]

    assert torch.allclose(torch.trace(h_original), torch.trace(h_rotated), atol=1e-5, rtol=1e-5)
    assert torch.allclose(
        torch.linalg.eigvalsh(h_original),
        torch.linalg.eigvalsh(h_rotated),
        atol=1e-5,
        rtol=1e-5,
    )
