import sys
from pathlib import Path

import torch
from e3nn import o3


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.so2_ops import SO2EdgeConv  # noqa: E402


def test_so2_edge_conv_is_equivariant_under_global_rotation():
    torch.manual_seed(33)
    irreps_in = o3.Irreps("4x0e + 4x1o + 4x2e")
    irreps_out = o3.Irreps("4x0e + 4x1o + 4x2e")
    conv = SO2EdgeConv(irreps_in, irreps_out)

    x = torch.randn(7, irreps_in.dim)
    edge_vec = torch.randn(7, 3)
    edge_vec = edge_vec + torch.tensor([0.2, -0.1, 0.3])
    weight = torch.randn(7, conv.weight_numel) * 0.1

    rotation = o3.rand_matrix().to(x.dtype)
    d_in = irreps_in.D_from_matrix(rotation).to(x.dtype)
    d_out = irreps_out.D_from_matrix(rotation).to(x.dtype)

    with torch.no_grad():
        out = conv(x, edge_vec, weight)
        out_rot = conv(x @ d_in.T, edge_vec @ rotation.T, weight)

    assert out.shape == (7, irreps_out.dim)
    assert torch.allclose(out_rot, out @ d_out.T, atol=1e-4, rtol=1e-4)


def test_so2_edge_conv_supports_reduced_hca_irreps():
    torch.manual_seed(34)
    irreps_in = o3.Irreps("4x0e + 4x1o + 4x2e + 4x3o + 4x4e")
    irreps_out = o3.Irreps("4x0e + 4x1o + 4x2e")
    conv = SO2EdgeConv(irreps_in, irreps_out)

    x = torch.randn(5, irreps_in.dim)
    edge_vec = torch.randn(5, 3)
    weight = torch.randn(5, conv.weight_numel) * 0.1
    out = conv(x, edge_vec, weight)

    assert out.shape == (5, irreps_out.dim)
    assert torch.isfinite(out).all()


def test_so2_edge_conv_mixes_nonzero_m_across_l_channels():
    torch.manual_seed(37)
    irreps_in = o3.Irreps("1x2e")
    irreps_out = o3.Irreps("1x1e")
    conv = SO2EdgeConv(irreps_in, irreps_out, cache_rotations=False)

    assert conv.weight_numel == 2

    x = torch.zeros(1, irreps_in.dim)
    x[:, 1] = 1.0  # contains a local |m|=1 component when edge_vec is z-aligned
    edge_vec = torch.tensor([[0.0, 0.0, 1.0]])
    weight = torch.tensor([[0.0, 1.0]])

    out = conv(x, edge_vec, weight)

    assert out.abs().max() > 1e-8


def test_so2_edge_conv_reuses_cached_rotation_for_identical_edges():
    torch.manual_seed(35)
    irreps = o3.Irreps("4x0e + 4x1o + 4x2e")
    conv = SO2EdgeConv(irreps, irreps)

    x = torch.randn(6, irreps.dim)
    edge_vec = torch.randn(6, 3)
    weight = torch.randn(6, conv.weight_numel) * 0.1

    out_first = conv(x, edge_vec, weight)
    assert conv.rotation_cache_misses == 1
    assert conv.rotation_cache_hits == 0
    assert conv.rotation_cache_size == 1

    out_second = conv(x, edge_vec.clone(), weight)
    assert conv.rotation_cache_misses == 1
    assert conv.rotation_cache_hits == 1
    assert conv.rotation_cache_size == 1
    assert torch.allclose(out_second, out_first, atol=1e-6, rtol=1e-6)


def test_so2_edge_conv_groups_cached_rotations_by_angular_order():
    torch.manual_seed(39)
    irreps = o3.Irreps("2x0e + 2x1o + 2x1e + 2x2e")
    conv = SO2EdgeConv(irreps, irreps)

    x = torch.randn(3, irreps.dim)
    edge_vec = torch.randn(3, 3)
    weight = torch.randn(3, conv.weight_numel) * 0.1

    conv(x, edge_vec, weight)
    entry = next(iter(conv._rotation_cache.values()))

    assert set(entry["input_d_by_l"]) == {0, 1, 2}
    assert set(entry["output_d_by_l"]) == {0, 1, 2}
    assert len(entry["input_d_by_l"]) < len(irreps)
    assert entry["input_d_by_l"][1].shape == (3, 3, 3)


def test_so2_edge_conv_uses_structured_m_channels_without_dense_projection_buffers():
    torch.manual_seed(40)
    irreps_in = o3.Irreps("2x0e + 2x1o + 2x2e")
    irreps_out = o3.Irreps("2x0e + 2x1o")
    conv = SO2EdgeConv(irreps_in, irreps_out)

    dense_projection_buffers = [
        name for name, _ in conv.named_buffers()
        if name.startswith("_proj_in_") or name.startswith("_proj_out_")
    ]

    assert dense_projection_buffers == []

    x = torch.randn(4, irreps_in.dim)
    edge_vec = torch.randn(4, 3)
    weight = torch.randn(4, conv.weight_numel) * 0.1
    out = conv(x, edge_vec, weight)

    assert out.shape == (4, irreps_out.dim)
    assert torch.isfinite(out).all()


def test_so2_edge_conv_organizes_channels_by_m_indices_like_deeptb():
    torch.manual_seed(41)
    irreps_in = o3.Irreps("2x0e + 2x1o + 2x2e")
    irreps_out = o3.Irreps("2x0e + 2x1o")
    conv = SO2EdgeConv(irreps_in, irreps_out)

    for spec_idx, spec in enumerate(conv.m_specs):
        m = spec["m"]
        in_index = getattr(conv, f"_m_in_index_{spec_idx}")
        out_index = getattr(conv, f"_m_out_index_{spec_idx}")

        expected_in = len(spec["in_channels"]) if m == 0 else 2 * len(spec["in_channels"])
        expected_out = len(spec["out_channels"]) if m == 0 else 2 * len(spec["out_channels"])
        assert in_index.dtype == torch.long
        assert out_index.dtype == torch.long
        assert in_index.numel() == expected_in
        assert out_index.numel() == expected_out

        if m > 0:
            assert torch.equal(in_index.reshape(-1, 2)[:, 1] - in_index.reshape(-1, 2)[:, 0], torch.ones(len(spec["in_channels"]), dtype=torch.long))
            assert torch.equal(out_index.reshape(-1, 2)[:, 1] - out_index.reshape(-1, 2)[:, 0], torch.ones(len(spec["out_channels"]), dtype=torch.long))


def test_so2_edge_conv_uses_m_native_packed_slices():
    torch.manual_seed(42)
    irreps_in = o3.Irreps("2x0e + 2x1o + 2x2e")
    irreps_out = o3.Irreps("2x0e + 2x1o")
    conv = SO2EdgeConv(irreps_in, irreps_out)

    in_pack = conv._m_in_pack_index
    out_pack = conv._m_out_pack_index
    assert in_pack.dtype == torch.long
    assert out_pack.dtype == torch.long

    in_cursor = 0
    out_cursor = 0
    for spec_idx, spec in enumerate(conv.m_specs):
        in_index = getattr(conv, f"_m_in_index_{spec_idx}")
        out_index = getattr(conv, f"_m_out_index_{spec_idx}")
        in_start, in_end = spec["in_slice"]
        out_start, out_end = spec["out_slice"]

        assert (in_start, in_end) == (in_cursor, in_cursor + in_index.numel())
        assert (out_start, out_end) == (out_cursor, out_cursor + out_index.numel())
        assert torch.equal(in_pack[in_start:in_end], in_index)
        assert torch.equal(out_pack[out_start:out_end], out_index)
        in_cursor = in_end
        out_cursor = out_end

    assert in_cursor == in_pack.numel()
    assert out_cursor == out_pack.numel()


def test_so2_edge_conv_uses_deeptb_style_linear_m_mixing(monkeypatch):
    torch.manual_seed(44)
    irreps = o3.Irreps("2x0e + 2x1o + 2x2e")
    conv = SO2EdgeConv(irreps, irreps, cache_rotations=False)

    x = torch.randn(4, irreps.dim)
    edge_vec = torch.randn(4, 3)
    weight = torch.randn(4, conv.weight_numel) * 0.1

    original_einsum = torch.einsum
    channel_mix_equations = []

    def traced_einsum(equation, *operands, **kwargs):
        if equation == "bc,oc->bo":
            channel_mix_equations.append(equation)
        return original_einsum(equation, *operands, **kwargs)

    monkeypatch.setattr(torch, "einsum", traced_einsum)

    out = conv(x, edge_vec, weight)

    assert torch.isfinite(out).all()
    assert channel_mix_equations == []


def test_so2_edge_conv_detached_rotation_path_does_not_force_cpu(monkeypatch):
    torch.manual_seed(45)
    irreps = o3.Irreps("2x0e + 2x1o + 2x2e")
    conv = SO2EdgeConv(irreps, irreps, cache_rotations=False, detach_rotations=True)

    x = torch.randn(4, irreps.dim)
    edge_vec = torch.randn(4, 3)
    weight = torch.randn(4, conv.weight_numel) * 0.1
    original_cpu = torch.Tensor.cpu

    def forbidden_cpu(tensor, *args, **kwargs):
        raise AssertionError("detached SO(2) rotation construction should stay on the input device")

    monkeypatch.setattr(torch.Tensor, "cpu", forbidden_cpu)
    try:
        out = conv(x, edge_vec, weight)
    finally:
        monkeypatch.setattr(torch.Tensor, "cpu", original_cpu)

    assert torch.isfinite(out).all()


def test_so2_edge_conv_device_safe_wigner_matches_e3nn_cpu():
    torch.manual_seed(46)
    irreps = o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o")
    conv = SO2EdgeConv(irreps, irreps)
    rotation = o3.rand_matrix(5)

    for l in range(irreps.lmax + 1):
        expected = o3.Irrep(l, 1).D_from_matrix(rotation)
        actual = conv._wigner_d_from_matrix(l, rotation)
        assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_so2_edge_conv_wigner_runtime_avoids_matrix_exp(monkeypatch):
    torch.manual_seed(47)
    irreps = o3.Irreps("1x0e + 1x1o + 1x2e + 1x3o")
    conv = SO2EdgeConv(irreps, irreps)
    rotation = o3.rand_matrix(5)
    original_matrix_exp = torch.matrix_exp

    def forbidden_matrix_exp(*args, **kwargs):
        raise AssertionError("runtime Wigner-D construction should use DeePTB-style sin/cos rotations")

    monkeypatch.setattr(torch, "matrix_exp", forbidden_matrix_exp)
    try:
        actual = conv._wigner_d_from_matrix(3, rotation)
    finally:
        monkeypatch.setattr(torch, "matrix_exp", original_matrix_exp)

    expected = o3.Irrep(3, 1).D_from_matrix(rotation)
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_so2_edge_conv_fuses_rotation_and_m_basis_transform():
    torch.manual_seed(43)
    irreps = o3.Irreps("2x0e + 2x1o + 2x2e")
    conv = SO2EdgeConv(irreps, irreps)

    x = torch.randn(5, irreps.dim)
    edge_vec = torch.randn(5, 3)
    rotation_entry = conv._get_rotation_cache_entry(edge_vec, dtype=x.dtype, device=x.device)

    local_in_blocks = conv._rotate_blocks(x, rotation_entry)
    two_step_in = conv._to_m_basis_flat(local_in_blocks, conv._input_l_groups)
    fused_in = conv._rotate_to_m_basis_flat(
        x,
        rotation_entry,
        conv.irreps_in,
        conv._input_l_groups,
        "input_d_by_l",
    )
    assert torch.allclose(fused_in, two_step_in, atol=1e-6, rtol=1e-6)

    canonical_out = torch.randn(5, irreps.dim)
    two_step_out = conv._rotate_out_blocks(
        conv._from_m_basis_flat(canonical_out, conv.irreps_out, conv._output_l_groups),
        rotation_entry,
    )
    fused_out = conv._rotate_from_m_basis_flat(
        canonical_out,
        rotation_entry,
        conv.irreps_out,
        conv._output_l_groups,
        "output_d_by_l",
    )
    assert torch.allclose(fused_out, two_step_out, atol=1e-6, rtol=1e-6)


def test_so2_edge_conv_rotation_cache_invalidates_when_edges_change():
    torch.manual_seed(36)
    irreps = o3.Irreps("4x0e + 4x1o")
    conv = SO2EdgeConv(irreps, irreps)

    x = torch.randn(5, irreps.dim)
    edge_vec = torch.randn(5, 3)
    changed_edge_vec = edge_vec.clone()
    changed_edge_vec[0, 0] = changed_edge_vec[0, 0] + 0.2
    weight = torch.randn(5, conv.weight_numel) * 0.1

    conv(x, edge_vec, weight)
    conv(x, changed_edge_vec, weight)

    assert conv.rotation_cache_misses == 2
    assert conv.rotation_cache_hits == 0
    assert conv.rotation_cache_size == 2


def test_so2_edge_conv_can_keep_edge_vector_gradients():
    torch.manual_seed(38)
    irreps = o3.Irreps("2x0e + 2x1o + 2x2e")
    conv = SO2EdgeConv(irreps, irreps, cache_rotations=False, detach_rotations=False)

    x = torch.randn(4, irreps.dim)
    edge_vec = torch.randn(4, 3, requires_grad=True)
    weight = torch.randn(4, conv.weight_numel) * 0.1
    loss = conv(x, edge_vec, weight).square().sum()
    loss.backward()

    assert edge_vec.grad is not None
    assert torch.isfinite(edge_vec.grad).all()
    assert edge_vec.grad.abs().max() > 0
