"""SO(2)-style edge-frame convolutions for e3nn irreps.

This QHformer operator follows the DeePTB-E3/EquiformerV2/eSCN idea:
rotate features to an edge-aligned frame, apply per-|m| SO(2)-equivariant
mixing, then rotate back.  Dense mixing weights are shared parameters; the
external edge weights act as radial/latent modulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from e3nn import o3
from e3nn.o3._wigner import so3_generators


def init_edge_frame(edge_vec, eps=1e-8):
    """Create deterministic right-handed frames with local z along edge_vec."""
    z_axis = edge_vec / edge_vec.norm(dim=-1, keepdim=True).clamp(min=eps)
    helper_z = torch.zeros_like(z_axis)
    helper_z[:, 2] = 1.0
    helper_y = torch.zeros_like(z_axis)
    helper_y[:, 1] = 1.0
    use_y = (z_axis * helper_z).sum(dim=-1, keepdim=True).abs() > 0.9
    helper = torch.where(use_y, helper_y, helper_z)

    x_axis = torch.cross(helper, z_axis, dim=-1)
    x_axis = x_axis / x_axis.norm(dim=-1, keepdim=True).clamp(min=eps)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    y_axis = y_axis / y_axis.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.stack([x_axis, y_axis, z_axis], dim=-1)


class SO2EdgeConv(nn.Module):
    """Edge-frame SO(2)-equivariant convolution with external radial modulation."""

    def __init__(
        self,
        irreps_in,
        irreps_out,
        match_parity=False,
        cache_rotations=True,
        max_rotation_cache_entries=8,
        detach_rotations=True,
    ):
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)
        self.match_parity = match_parity
        self.cache_rotations = cache_rotations
        self.detach_rotations = detach_rotations
        self.max_rotation_cache_entries = max(1, int(max_rotation_cache_entries))
        self._rotation_cache = OrderedDict()
        self._rotation_cache_hits = 0
        self._rotation_cache_misses = 0
        self._register_m_bases()
        self._register_l_transforms()
        self._register_axis_rotation_bases()
        self._input_l_groups = self._build_l_groups(self.irreps_in)
        self._output_l_groups = self._build_l_groups(self.irreps_out)

        self.front = self.irreps_in.dim <= self.irreps_out.dim
        in_channels_by_m = self._build_m_channels(self.irreps_in)
        out_channels_by_m = self._build_m_channels(self.irreps_out)
        parity_groups = (-1, 1) if match_parity else (None,)
        self.m_specs = []
        self.mixing_weights = nn.ParameterList()
        weight_numel = 0
        for m in range(min(self.irreps_in.lmax, self.irreps_out.lmax) + 1):
            for parity in parity_groups:
                in_channels = self._filter_parity(in_channels_by_m[m], parity)
                out_channels = self._filter_parity(out_channels_by_m[m], parity)
                if not in_channels or not out_channels:
                    continue
                radial_size = len(in_channels) if self.front else len(out_channels)
                param_shape = (
                    (len(out_channels), len(in_channels))
                    if m == 0
                    else (2 * len(out_channels), len(in_channels))
                )
                mixing_weight = nn.Parameter(torch.empty(*param_shape))
                self._init_mixing_weight(mixing_weight, m)
                self.mixing_weights.append(mixing_weight)
                self.m_specs.append({
                    "m": m,
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "in_groups": self._channels_to_block_groups(self.irreps_in, in_channels),
                    "out_groups": self._channels_to_block_groups(self.irreps_out, out_channels),
                    "offset": weight_numel,
                    "radial_size": radial_size,
                })
                weight_numel += radial_size
        if weight_numel == 0:
            raise ValueError(f"No SO(2) paths from {self.irreps_in} to {self.irreps_out}")
        self.weight_numel = weight_numel
        self._register_m_indices()

    @staticmethod
    def _build_m_channels(irreps):
        channels = [[] for _ in range(irreps.lmax + 1)]
        for block_idx, (mul, ir) in enumerate(irreps):
            for mul_idx in range(mul):
                for m in range(ir.l + 1):
                    channels[m].append((block_idx, mul_idx, ir.l, ir.p))
        return channels

    @staticmethod
    def _filter_parity(channels, parity):
        if parity is None:
            return channels
        return [channel for channel in channels if channel[3] == parity]

    @staticmethod
    def _channels_to_block_groups(irreps, channels):
        groups = []
        i = 0
        while i < len(channels):
            block_idx, _, l, _ = channels[i]
            mul, _ = irreps[block_idx]
            mul_indices = []
            while i < len(channels) and channels[i][0] == block_idx:
                mul_indices.append(channels[i][1])
                i += 1
            if mul_indices != list(range(int(mul))):
                raise ValueError("SO2EdgeConv expects complete multiplicity groups per irrep block")
            groups.append({
                "block_idx": block_idx,
                "l": l,
                "mul": int(mul),
            })
        return groups

    @staticmethod
    def _channel_flat_offset(irreps, channel):
        block_idx, mul_idx, l, _ = channel
        slc = irreps.slices()[block_idx]
        return slc.start + mul_idx * (2 * l + 1)

    @staticmethod
    def _init_mixing_weight(weight, m):
        with torch.no_grad():
            if m == 0:
                nn.init.xavier_uniform_(weight)
            else:
                out_channels = weight.shape[0] // 2
                nn.init.xavier_uniform_(weight[:out_channels])
                nn.init.xavier_uniform_(weight[out_channels:])
                weight.mul_(1.0 / math.sqrt(2.0))

    @staticmethod
    def _z_rotation(theta, *, dtype=torch.float64):
        c = torch.cos(torch.as_tensor(theta, dtype=dtype))
        s = torch.sin(torch.as_tensor(theta, dtype=dtype))
        zero = torch.zeros((), dtype=dtype)
        one = torch.ones((), dtype=dtype)
        return torch.stack([
            torch.stack([c, -s, zero]),
            torch.stack([s, c, zero]),
            torch.stack([zero, zero, one]),
        ])

    @staticmethod
    def _fix_basis_sign(vector):
        idx = vector.abs().argmax()
        if vector[idx] < 0:
            vector = -vector
        return vector

    @classmethod
    def _canonical_z_m_basis(cls, l, m):
        if l == 0:
            return torch.ones(1, dtype=torch.float64)

        eps = 1e-5
        ir = o3.Irrep(l, 1)
        d_plus = ir.D_from_matrix(cls._z_rotation(eps)).to(torch.float64)
        d_minus = ir.D_from_matrix(cls._z_rotation(-eps)).to(torch.float64)
        generator = (d_plus - d_minus) / (2.0 * eps)
        frequency_operator = -(generator @ generator)
        evals, evecs = torch.linalg.eigh(frequency_operator)
        target = float(m * m)
        indices = torch.nonzero(torch.isclose(evals, torch.full_like(evals, target), atol=1e-4), as_tuple=False).flatten()
        if m == 0:
            if indices.numel() != 1:
                raise RuntimeError(f"Expected one m=0 vector for l={l}, found {indices.numel()}")
            return cls._fix_basis_sign(evecs[:, indices[0]])

        if indices.numel() != 2:
            raise RuntimeError(f"Expected two |m|={m} vectors for l={l}, found {indices.numel()}")
        subspace = evecs[:, indices]
        u = cls._fix_basis_sign(subspace[:, 0])
        v = generator @ u / float(m)
        v = v - u * torch.dot(u, v)
        v = v / v.norm().clamp(min=1e-12)
        return torch.stack([u, v], dim=0)

    def _register_m_bases(self):
        l_values = sorted({ir.l for _, ir in self.irreps_in} | {ir.l for _, ir in self.irreps_out})
        for l in l_values:
            for m in range(l + 1):
                self.register_buffer(f"_basis_l{l}_m{m}", self._canonical_z_m_basis(l, m).to(torch.float32))

    def _register_l_transforms(self):
        l_values = sorted({ir.l for _, ir in self.irreps_in} | {ir.l for _, ir in self.irreps_out})
        for l in l_values:
            rows = []
            for m in range(l + 1):
                basis = getattr(self, f"_basis_l{l}_m{m}")
                rows.append(basis.unsqueeze(0) if m == 0 else basis)
            self.register_buffer(f"_transform_l{l}", torch.cat(rows, dim=0))

    @classmethod
    def _axis_rotation_basis(cls, l, axis):
        if l == 0:
            return torch.ones(1, 1, dtype=torch.float64)

        generator = so3_generators(l).to(torch.float64)[axis]
        frequency_operator = -(generator @ generator)
        evals, evecs = torch.linalg.eigh(frequency_operator)
        rows = []
        for m in range(l + 1):
            target = float(m * m)
            indices = torch.nonzero(
                torch.isclose(evals, torch.full_like(evals, target), atol=1e-4),
                as_tuple=False,
            ).flatten()
            if m == 0:
                if indices.numel() != 1:
                    raise RuntimeError(f"Expected one axis m=0 vector for l={l}, axis={axis}, found {indices.numel()}")
                rows.append(cls._fix_basis_sign(evecs[:, indices[0]]))
                continue

            if indices.numel() != 2:
                raise RuntimeError(f"Expected two axis |m|={m} vectors for l={l}, axis={axis}, found {indices.numel()}")
            subspace = evecs[:, indices]
            u = cls._fix_basis_sign(subspace[:, 0])
            v = generator @ u / float(m)
            v = v - u * torch.dot(u, v)
            v = v / v.norm().clamp(min=1e-12)
            rows.extend([u, v])
        return torch.stack(rows, dim=0)

    def _register_axis_rotation_bases(self):
        l_values = sorted({ir.l for _, ir in self.irreps_in} | {ir.l for _, ir in self.irreps_out})
        for l in l_values:
            for axis in (0, 1):
                self.register_buffer(
                    f"_axis_basis_l{l}_a{axis}",
                    self._axis_rotation_basis(l, axis).to(torch.float32),
                )

    @staticmethod
    def _build_l_groups(irreps):
        groups = {}
        for block_idx, (slc, (mul, ir)) in enumerate(zip(irreps.slices(), irreps)):
            groups.setdefault(ir.l, []).append({
                "block_idx": block_idx,
                "slice": slc,
                "mul": mul,
                "dim": ir.dim,
            })
        return groups

    def _m_basis(self, l, m, *, dtype, device):
        return getattr(self, f"_basis_l{l}_m{m}").to(dtype=dtype, device=device)

    def _l_transform(self, l, *, dtype, device):
        return getattr(self, f"_transform_l{l}").to(dtype=dtype, device=device)

    def _wigner_d_from_matrix(self, l, rotation_matrix):
        alpha, beta, gamma = o3.matrix_to_angles(rotation_matrix)
        alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
        return (
            self._axis_rotation(l, 1, alpha)
            @ self._axis_rotation(l, 0, beta)
            @ self._axis_rotation(l, 1, gamma)
        )

    def _axis_rotation(self, l, axis, angle):
        basis = getattr(self, f"_axis_basis_l{l}_a{axis}").to(
            dtype=angle.dtype,
            device=angle.device,
        )
        dim = 2 * l + 1
        local = angle.new_zeros((*angle.shape, dim, dim))
        local[..., 0, 0] = 1.0
        for m in range(1, l + 1):
            offset = 2 * m - 1
            c = torch.cos(float(m) * angle)
            s = torch.sin(float(m) * angle)
            local[..., offset, offset] = c
            local[..., offset, offset + 1] = -s
            local[..., offset + 1, offset] = s
            local[..., offset + 1, offset + 1] = c
        return basis.transpose(0, 1) @ local @ basis

    @staticmethod
    def _m_offset(m):
        return 0 if m == 0 else 2 * m - 1

    def _m_channel_indices(self, irreps, channels, m):
        indices = []
        offset = self._m_offset(m)
        for channel in channels:
            _, _, l, _ = channel
            start = self._channel_flat_offset(irreps, channel)
            if m == 0:
                indices.append(start + offset)
            else:
                indices.extend([start + offset, start + offset + 1])
        return torch.tensor(indices, dtype=torch.long)

    def _register_m_indices(self):
        in_pack_indices = []
        out_pack_indices = []
        in_offset = 0
        out_offset = 0
        for spec_idx, spec in enumerate(self.m_specs):
            m = spec["m"]
            in_index = self._m_channel_indices(self.irreps_in, spec["in_channels"], m)
            out_index = self._m_channel_indices(self.irreps_out, spec["out_channels"], m)
            self.register_buffer(
                f"_m_in_index_{spec_idx}",
                in_index,
            )
            self.register_buffer(
                f"_m_out_index_{spec_idx}",
                out_index,
            )
            spec["in_slice"] = (in_offset, in_offset + in_index.numel())
            spec["out_slice"] = (out_offset, out_offset + out_index.numel())
            in_pack_indices.append(in_index)
            out_pack_indices.append(out_index)
            in_offset += in_index.numel()
            out_offset += out_index.numel()
        self.register_buffer("_m_in_pack_index", torch.cat(in_pack_indices))
        self.register_buffer("_m_out_pack_index", torch.cat(out_pack_indices))
        self._m_in_pack_dim = in_offset
        self._m_out_pack_dim = out_offset

    @property
    def rotation_cache_hits(self):
        return self._rotation_cache_hits

    @property
    def rotation_cache_misses(self):
        return self._rotation_cache_misses

    @property
    def rotation_cache_size(self):
        return len(self._rotation_cache)

    def clear_rotation_cache(self, reset_stats=False):
        self._rotation_cache.clear()
        if reset_stats:
            self._rotation_cache_hits = 0
            self._rotation_cache_misses = 0

    def _apply(self, fn):
        self.clear_rotation_cache()
        return super()._apply(fn)

    def _cache_entry_matches(self, entry, edge_vec):
        cached_edge_vec = entry["edge_vec"]
        return (
            cached_edge_vec.shape == edge_vec.shape
            and cached_edge_vec.device == edge_vec.device
            and cached_edge_vec.dtype == edge_vec.dtype
            and torch.equal(cached_edge_vec, edge_vec)
        )

    def _build_rotation_cache_entry(self, edge_vec):
        frame = init_edge_frame(edge_vec)
        inverse_frame = frame.transpose(-1, -2)
        input_l_values = sorted(self._input_l_groups)
        output_l_values = sorted(self._output_l_groups)
        if self.detach_rotations:
            inverse_frame_for_d = inverse_frame.detach()
            frame_for_d = frame.detach()
            input_d_by_l = {
                l: self._wigner_d_from_matrix(l, inverse_frame_for_d).to(dtype=edge_vec.dtype, device=edge_vec.device).detach()
                for l in input_l_values
            }
            output_d_by_l = {
                l: self._wigner_d_from_matrix(l, frame_for_d).to(dtype=edge_vec.dtype, device=edge_vec.device).detach()
                for l in output_l_values
            }
        else:
            input_d_by_l = {
                l: self._wigner_d_from_matrix(l, inverse_frame).to(dtype=edge_vec.dtype, device=edge_vec.device)
                for l in input_l_values
            }
            output_d_by_l = {
                l: self._wigner_d_from_matrix(l, frame).to(dtype=edge_vec.dtype, device=edge_vec.device)
                for l in output_l_values
            }
        return {
            "edge_vec": edge_vec.detach().clone(),
            "input_d_by_l": input_d_by_l,
            "output_d_by_l": output_d_by_l,
        }

    def _get_rotation_cache_entry(self, edge_vec, *, dtype, device):
        edge_vec = edge_vec.to(dtype=dtype, device=device)
        if not self.cache_rotations or not self.detach_rotations:
            return self._build_rotation_cache_entry(edge_vec)

        edge_vec_key = edge_vec.detach()
        for cache_key, entry in list(self._rotation_cache.items()):
            if self._cache_entry_matches(entry, edge_vec_key):
                self._rotation_cache_hits += 1
                self._rotation_cache.move_to_end(cache_key)
                return entry

        self._rotation_cache_misses += 1
        entry = self._build_rotation_cache_entry(edge_vec)
        cache_key = self._rotation_cache_misses
        self._rotation_cache[cache_key] = entry
        while len(self._rotation_cache) > self.max_rotation_cache_entries:
            self._rotation_cache.popitem(last=False)
        return entry

    def _rotate_blocks(self, x, rotation_entry):
        blocks = [None] * len(self.irreps_in)
        for l, group in self._input_l_groups.items():
            d_matrix = rotation_entry["input_d_by_l"][l]
            local = torch.cat([
                x[:, item["slice"]].reshape(x.shape[0], item["mul"], item["dim"])
                for item in group
            ], dim=1)
            rotated = torch.einsum("bmd,bkd->bmk", local, d_matrix)
            offset = 0
            for item in group:
                next_offset = offset + item["mul"]
                blocks[item["block_idx"]] = rotated[:, offset:next_offset]
                offset = next_offset
        return blocks

    def _rotate_out_blocks(self, blocks, rotation_entry):
        pieces = [None] * len(blocks)
        for l, group in self._output_l_groups.items():
            d_matrix = rotation_entry["output_d_by_l"][l]
            local = torch.cat([blocks[item["block_idx"]] for item in group], dim=1)
            rotated = torch.einsum("bmd,bkd->bmk", local, d_matrix)
            offset = 0
            for item in group:
                next_offset = offset + item["mul"]
                pieces[item["block_idx"]] = rotated[:, offset:next_offset].reshape(rotated.shape[0], -1)
                offset = next_offset
        return torch.cat(pieces, dim=-1)

    def _to_m_basis_blocks(self, local_blocks, l_groups):
        canonical_blocks = [None] * len(local_blocks)
        for l, group in l_groups.items():
            transform = self._l_transform(l, dtype=local_blocks[group[0]["block_idx"]].dtype, device=local_blocks[group[0]["block_idx"]].device)
            for item in group:
                block = local_blocks[item["block_idx"]]
                canonical_blocks[item["block_idx"]] = torch.einsum("bmd,kd->bmk", block, transform)
        return canonical_blocks

    @staticmethod
    def _flatten_blocks(blocks):
        return torch.cat([block.reshape(block.shape[0], -1) for block in blocks], dim=-1)

    @staticmethod
    def _unflatten_blocks(flat, irreps):
        return [
            flat[:, slc].reshape(flat.shape[0], mul, ir.dim)
            for slc, (mul, ir) in zip(irreps.slices(), irreps)
        ]

    def _to_m_basis_flat(self, local_blocks, l_groups):
        return self._flatten_blocks(self._to_m_basis_blocks(local_blocks, l_groups))

    def _rotate_to_m_basis_flat(self, x, rotation_entry, irreps, l_groups, d_key):
        pieces = [None] * len(irreps)
        for l, group in l_groups.items():
            d_matrix = rotation_entry[d_key][l]
            transform = self._l_transform(l, dtype=x.dtype, device=x.device)
            fused = torch.einsum("kr,brd->bkd", transform, d_matrix)
            local = torch.cat([
                x[:, item["slice"]].reshape(x.shape[0], item["mul"], item["dim"])
                for item in group
            ], dim=1)
            canonical = torch.einsum("bmd,bkd->bmk", local, fused)
            offset = 0
            for item in group:
                next_offset = offset + item["mul"]
                pieces[item["block_idx"]] = canonical[:, offset:next_offset].reshape(x.shape[0], -1)
                offset = next_offset
        return torch.cat(pieces, dim=-1)

    def _from_m_basis_blocks(self, canonical_blocks, l_groups):
        local_blocks = [None] * len(canonical_blocks)
        for l, group in l_groups.items():
            transform = self._l_transform(l, dtype=canonical_blocks[group[0]["block_idx"]].dtype, device=canonical_blocks[group[0]["block_idx"]].device)
            for item in group:
                block = canonical_blocks[item["block_idx"]]
                local_blocks[item["block_idx"]] = torch.einsum("bmk,kd->bmd", block, transform)
        return local_blocks

    def _from_m_basis_flat(self, canonical_flat, irreps, l_groups):
        canonical_blocks = self._unflatten_blocks(canonical_flat, irreps)
        return self._from_m_basis_blocks(canonical_blocks, l_groups)

    def _rotate_from_m_basis_flat(self, canonical_flat, rotation_entry, irreps, l_groups, d_key):
        canonical_blocks = self._unflatten_blocks(canonical_flat, irreps)
        pieces = [None] * len(irreps)
        for l, group in l_groups.items():
            d_matrix = rotation_entry[d_key][l]
            transform = self._l_transform(l, dtype=canonical_flat.dtype, device=canonical_flat.device)
            fused = torch.einsum("bkd,jd->bkj", d_matrix, transform)
            local = torch.cat([canonical_blocks[item["block_idx"]] for item in group], dim=1)
            rotated = torch.einsum("bmj,bkj->bmk", local, fused)
            offset = 0
            for item in group:
                next_offset = offset + item["mul"]
                pieces[item["block_idx"]] = rotated[:, offset:next_offset].reshape(canonical_flat.shape[0], -1)
                offset = next_offset
        return torch.cat(pieces, dim=-1)

    def _project_m(self, canonical_packed, spec):
        m = spec["m"]
        in_start, in_end = spec["in_slice"]
        gathered = canonical_packed[:, in_start:in_end]
        if m == 0:
            return gathered
        return gathered.reshape(canonical_packed.shape[0], -1, 2).transpose(1, 2).contiguous()

    def _add_m_to_packed(self, canonical_out_packed, values, spec):
        m = spec["m"]
        if m > 0:
            values = values.transpose(1, 2).contiguous().reshape(values.shape[0], -1)
        out_start, out_end = spec["out_slice"]
        canonical_out_packed[:, out_start:out_end] = (
            canonical_out_packed[:, out_start:out_end] + values
        )

    @staticmethod
    def _mix_nonzero_m(local_m, mixing_weight):
        out_channels = mixing_weight.shape[0] // 2
        linear_output = F.linear(local_m, mixing_weight)
        real_projection = linear_output.narrow(2, 0, out_channels)
        imag_projection = linear_output.narrow(2, out_channels, out_channels)
        real_out = real_projection.narrow(1, 0, 1) - imag_projection.narrow(1, 1, 1)
        imag_out = real_projection.narrow(1, 1, 1) + imag_projection.narrow(1, 0, 1)
        return torch.cat((real_out, imag_out), dim=1)

    def _pack_m_channels(self, canonical_flat):
        return canonical_flat.index_select(
            1,
            self._m_in_pack_index.to(device=canonical_flat.device),
        )

    def _unpack_m_channels(self, canonical_packed, out_dim):
        canonical_flat = canonical_packed.new_zeros(canonical_packed.shape[0], out_dim)
        canonical_flat.index_copy_(
            1,
            self._m_out_pack_index.to(device=canonical_packed.device),
            canonical_packed,
        )
        return canonical_flat

    def forward(self, x, edge_vec, weight):
        if x.shape[-1] != self.irreps_in.dim:
            raise ValueError(f"Expected input dim {self.irreps_in.dim}, got {x.shape[-1]}")
        if weight.shape[-1] != self.weight_numel:
            raise ValueError(f"Expected weight dim {self.weight_numel}, got {weight.shape[-1]}")

        batch = x.shape[0]
        rotation_entry = self._get_rotation_cache_entry(edge_vec, dtype=x.dtype, device=x.device)
        canonical_in_flat = self._rotate_to_m_basis_flat(
            x,
            rotation_entry,
            self.irreps_in,
            self._input_l_groups,
            "input_d_by_l",
        )
        canonical_in_packed = self._pack_m_channels(canonical_in_flat)
        canonical_out_packed = x.new_zeros(batch, self._m_out_pack_dim)

        for spec, mixing_weight in zip(self.m_specs, self.mixing_weights):
            m = spec["m"]
            radial = weight[:, spec["offset"]:spec["offset"] + spec["radial_size"]]
            local_m = self._project_m(canonical_in_packed, spec)
            if self.front:
                local_m = local_m * radial if m == 0 else local_m * radial.unsqueeze(1)

            if m == 0:
                mixed = F.linear(local_m, mixing_weight)
                if not self.front:
                    mixed = mixed * radial
            else:
                mixed = self._mix_nonzero_m(local_m, mixing_weight)
                if not self.front:
                    mixed = mixed * radial.unsqueeze(1)
            self._add_m_to_packed(canonical_out_packed, mixed, spec)

        canonical_out_flat = self._unpack_m_channels(canonical_out_packed, self.irreps_out.dim)
        return self._rotate_from_m_basis_flat(
            canonical_out_flat,
            rotation_entry,
            self.irreps_out,
            self._output_l_groups,
            "output_d_by_l",
        )
