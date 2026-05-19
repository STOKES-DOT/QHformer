"""
Inner Product Attention for Equivariant Neural Networks

This module implements Scheme 1: Inner Product Attention
The key idea is to use InnerProduct to couple irreps to scalars for attention weight computation.

Mathematical Principle:
    Given query q_i ∈ V^(l1) ⊕ V^(l2) ⊕ ...
          key   k_j ∈ V^(l1) ⊕ V^(l2) ⊕ ...

    InnerProduct: IP(q_i, k_j) = Σ_l Σ_m q_i^(l,m) · k_j^(l,m)  → scalar

    Attention weight: α_ij = softmax(IP(q_i, k_j) / √d)

    This maintains SO(3) equivariance because the inner product of
    two irreps of the same degree is rotationally invariant.

Reference: e3nn transformer attention mechanism
"""

import torch
import torch.nn as nn
import math
from e3nn import o3
from e3nn.o3 import Linear, TensorProduct
from e3nn.nn import FullyConnectedNet

try:
    from .torch_ops import scatter
except ImportError:
    from torch_ops import scatter


def get_feasible_irrep(irrep_in1, irrep_in2, cutoff_irrep_out, tp_mode="uvu"):
    """
    Get feasible irreducible representations for tensor product.

    Args:
        irrep_in1: First input irreps
        irrep_in2: Second input irreps
        cutoff_irrep_out: Output irreps to filter by
        tp_mode: Tensor product mode ('uvw', 'uvu', 'uvv', etc.)

    Returns:
        irrep_mid: Intermediate irreps
        instructions: Tensor product instructions
    """
    irrep_mid = []
    instructions = []

    for i, (_, ir_in) in enumerate(irrep_in1):
        for j, (_, ir_edge) in enumerate(irrep_in2):
            for ir_out in ir_in * ir_edge:
                if ir_out in cutoff_irrep_out:
                    if (cutoff_irrep_out.count(ir_out), ir_out) not in irrep_mid:
                        k = len(irrep_mid)
                        irrep_mid.append((cutoff_irrep_out.count(ir_out), ir_out))
                    else:
                        k = irrep_mid.index((cutoff_irrep_out.count(ir_out), ir_out))
                    instructions.append((i, j, k, tp_mode, True))

    irrep_mid = o3.Irreps(irrep_mid)
    normalization_coefficients = []
    for ins in instructions:
        ins_dict = {
            'uvw': (irrep_in1[ins[0]].mul * irrep_in2[ins[1]].mul),
            'uvu': irrep_in2[ins[1]].mul,
            'uvv': irrep_in1[ins[0]].mul,
            'uuw': irrep_in1[ins[0]].mul,
            'uuu': 1,
        }
        alpha = irrep_mid[ins[2]].ir.dim
        x = sum([ins_dict[ins[3]] for ins in instructions])
        if x > 0.0:
            alpha /= x
        normalization_coefficients += [math.sqrt(alpha)]

    irrep_mid, p, _ = irrep_mid.sort()
    instructions = [
        (i_in1, i_in2, p[i_out], mode, train, alpha)
        for (i_in1, i_in2, i_out, mode, train), alpha
        in zip(instructions, normalization_coefficients)
    ]
    return irrep_mid, instructions


class InnerProduct(nn.Module):
    """
    Inner product for computing invariant features from irreps.

    Computes: IP(x, y) = Σ_l x^(l) · y^(l)  → scalar

    This is SO(3) equivariant because the inner product of two
    irreps of the same degree l is rotationally invariant.
    """

    def __init__(self, irrep_in):
        super(InnerProduct, self).__init__()
        self.irrep_in = o3.Irreps(irrep_in).simplify()
        # Output is scalar (0e) for each input irrep
        irrep_out = o3.Irreps([(mul, "0e") for mul, _ in self.irrep_in])
        instr = [(i, i, i, "uuu", False, 1/ir.dim) for i, (mul, ir) in enumerate(self.irrep_in)]
        self.tp = o3.TensorProduct(
            self.irrep_in, self.irrep_in, irrep_out, instr,
            irrep_normalization="component"
        )
        self.irrep_out = irrep_out.simplify()

    def forward(self, features_1, features_2):
        """
        Compute inner product: IP(features_1, features_2)

        Args:
            features_1: [N, D_in] or [E, D_in]
            features_2: [N, D_in] or [E, D_in]

        Returns:
            out: [N, num_scalars] or [E, num_scalars]
        """
        out = self.tp(features_1, features_2)
        return out


class EquivariantNorm(nn.Module):
    """
    Equivariant normalization layer.

    Normalizes each irrep channel by its norm, preserving equivariance.
    For scalar (l=0): standard normalization
    For higher-order (l>0): divide by norm to get unit vectors

    This prevents numerical overflow in InnerProduct computations.
    """

    def __init__(self, irreps, eps=1e-8):
        super(EquivariantNorm, self).__init__()
        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.norm = o3.Norm(self.irreps)

    def forward(self, x):
        """
        Args:
            x: [N, D] equivariant features

        Returns:
            normalized_x: [N, D] normalized features (same scale)
        """
        # Compute norm for each irrep
        # IMPORTANT: o3.Norm outputs [N, total_mul], not [N, num_irreps]!
        # Each multiplicity gets its own norm value.
        norms = self.norm(x)  # [N, total_mul] where total_mul = sum(mul for mul, ir in irreps)

        # Normalize: divide each irrep by its norm
        normalized_x = x.clone()
        norm_idx = 0  # Track position in the norms tensor

        for i, (mul, ir) in enumerate(self.irreps):
            # Get slice indices for features
            start = self.irreps.slices()[i].start
            end = self.irreps.slices()[i].stop

            # MATHEMATICAL FIX: Extract correct slice of norms for this irrep block
            # Since norms has shape [N, total_mul], we need to extract exactly 'mul' norms
            # for this irrep, NOT just one norm to broadcast.
            block_norms = norms[:, norm_idx : norm_idx + mul]  # [N, mul]
            norm_idx += mul  # Increment counter by mul (not by 1!)

            # Extract corresponding feature block
            block_x = x[:, start:end]  # [N, mul * ir.dim]

            # Reshape for proper broadcasting
            block_x = block_x.reshape(-1, mul, ir.dim)  # [N, mul, ir.dim]
            block_norms = block_norms.reshape(-1, mul, 1)  # [N, mul, 1]

            # Safely divide (apply clamp to avoid division by zero)
            normalized_block = block_x / block_norms.clamp(min=self.eps)

            # Reshape back and assign to output
            normalized_x[:, start:end] = normalized_block.reshape(-1, mul * ir.dim)

        return normalized_x


def _per_head_irrep(irreps, num_heads):
    """Split multiplicities evenly across heads without splitting irrep dimensions."""
    irreps = o3.Irreps(irreps)
    per_head = []
    for mul, ir in irreps:
        if mul % num_heads != 0:
            raise ValueError(
                f"Irrep multiplicity {mul} for {ir} is not divisible by "
                f"num_heads={num_heads}"
            )
        per_head.append((mul // num_heads, ir))
    return o3.Irreps(per_head).simplify()


def _filter_irrep_lmax(irreps, lmax):
    """Keep only irreps with angular degree l <= lmax."""
    irreps = o3.Irreps(irreps)
    return o3.Irreps([(mul, ir) for mul, ir in irreps if ir.l <= lmax]).simplify()


def split_irreps_multiplicity(x, irreps, num_heads):
    """Split a flat irrep tensor into heads along the multiplicity axis."""
    irreps = o3.Irreps(irreps)
    if x.shape[-1] != irreps.dim:
        raise ValueError(f"Expected last dim {irreps.dim}, got {x.shape[-1]}")

    pieces = []
    batch = x.shape[0]
    for slc, (mul, ir) in zip(irreps.slices(), irreps):
        if mul % num_heads != 0:
            raise ValueError(
                f"Irrep multiplicity {mul} for {ir} is not divisible by "
                f"num_heads={num_heads}"
            )
        mul_per_head = mul // num_heads
        block = x[:, slc].reshape(batch, num_heads, mul_per_head, ir.dim)
        pieces.append(block.reshape(batch, num_heads, mul_per_head * ir.dim))
    return torch.cat(pieces, dim=-1)


def merge_heads(x, irreps, num_heads):
    """Reverse ``split_irreps_multiplicity``."""
    irreps = o3.Irreps(irreps)
    if x.dim() != 3:
        raise ValueError(f"Expected [N, H, D_head], got shape {tuple(x.shape)}")
    if x.shape[1] != num_heads:
        raise ValueError(f"Expected {num_heads} heads, got {x.shape[1]}")

    pieces = []
    start = 0
    batch = x.shape[0]
    for mul, ir in irreps:
        if mul % num_heads != 0:
            raise ValueError(
                f"Irrep multiplicity {mul} for {ir} is not divisible by "
                f"num_heads={num_heads}"
            )
        mul_per_head = mul // num_heads
        width = mul_per_head * ir.dim
        block = x[:, :, start:start + width]
        block = block.reshape(batch, num_heads, mul_per_head, ir.dim)
        pieces.append(block.reshape(batch, mul * ir.dim))
        start += width

    if start != x.shape[-1]:
        raise ValueError(f"Unused head channels: consumed {start}, got {x.shape[-1]}")
    return torch.cat(pieces, dim=-1)


def filter_irrep_tensor_lmax(x, irreps, lmax):
    """Extract all flat irrep blocks with l <= lmax."""
    irreps = o3.Irreps(irreps)
    pieces = [x[:, slc] for slc, (_, ir) in zip(irreps.slices(), irreps) if ir.l <= lmax]
    if not pieces:
        return x.new_zeros(x.shape[0], 0)
    return torch.cat(pieces, dim=-1)


def pad_irrep_tensor(x, source_irreps, target_irreps):
    """Pad a reduced irrep tensor with zeros to match target irreps."""
    source_irreps = o3.Irreps(source_irreps)
    target_irreps = o3.Irreps(target_irreps)
    source_by_ir = {}
    for slc, (mul, ir) in zip(source_irreps.slices(), source_irreps):
        source_by_ir[(ir.l, ir.p)] = (slc, mul, ir)

    pieces = []
    for mul, ir in target_irreps:
        source = source_by_ir.get((ir.l, ir.p))
        if source is None:
            pieces.append(x.new_zeros(x.shape[0], mul * ir.dim))
            continue
        slc, source_mul, _ = source
        if source_mul != mul:
            raise ValueError(
                f"Cannot pad {source_irreps} to {target_irreps}: "
                f"multiplicity mismatch for {ir}"
            )
        pieces.append(x[:, slc])
    return torch.cat(pieces, dim=-1)


class MultiHeadInnerProduct(nn.Module):
    """Per-head invariant inner product over matching irreps."""

    def __init__(self, irrep_in, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irrep_in = o3.Irreps(irrep_in).simplify()
        self.per_head_irrep = _per_head_irrep(self.irrep_in, num_heads)
        self.inner_product = InnerProduct(self.per_head_irrep)
        self.irrep_out = self.inner_product.irrep_out

    def forward(self, features_1, features_2):
        if features_1.shape != features_2.shape:
            raise ValueError(
                f"Multi-head inner product requires equal shapes, got "
                f"{tuple(features_1.shape)} and {tuple(features_2.shape)}"
            )
        if features_1.dim() != 3 or features_1.shape[1] != self.num_heads:
            raise ValueError(
                f"Expected [N/E, {self.num_heads}, D_head], got "
                f"{tuple(features_1.shape)}"
            )

        n = features_1.shape[0]
        flat_1 = features_1.reshape(n * self.num_heads, -1)
        flat_2 = features_2.reshape(n * self.num_heads, -1)
        out = self.inner_product(flat_1, flat_2)
        out = out.reshape(n, self.num_heads, -1)
        return out.transpose(1, 2).contiguous()


class InvariantAttentionScore(nn.Module):
    """
    Learnable scorer over invariant inner-product channels.

    The base term is the previous fixed inner-product sum.  The learnable
    residual starts at exactly zero, so initialization preserves existing
    attention behavior while allowing training to mix invariant channels.
    """

    def __init__(self, num_invariants, num_heads, hidden_dim=32, residual_init_std=0.0):
        super().__init__()
        self.num_invariants = num_invariants
        self.num_heads = num_heads
        self.residual_init_std = residual_init_std
        self.input = nn.Linear(num_invariants, hidden_dim)
        self.activation = nn.SiLU()
        self.output = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.input.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.input.bias)
        if self.residual_init_std > 0:
            nn.init.normal_(self.output.weight, mean=0.0, std=self.residual_init_std)
        else:
            nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, attention_input):
        if attention_input.dim() != 3:
            raise ValueError(
                f"Expected [E, num_invariants, num_heads], got {tuple(attention_input.shape)}"
            )
        if attention_input.shape[1] != self.num_invariants:
            raise ValueError(
                f"Expected {self.num_invariants} invariant channels, got {attention_input.shape[1]}"
            )
        if attention_input.shape[2] != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} heads, got {attention_input.shape[2]}")

        base = attention_input.sum(dim=1)
        num_edges = attention_input.shape[0]
        flat = attention_input.transpose(1, 2).reshape(num_edges * self.num_heads, self.num_invariants)
        residual = self.output(self.activation(self.input(flat))).reshape(num_edges, self.num_heads)
        return base + residual


class MultiHeadEquivariantNorm(nn.Module):
    """Equivariant normalization applied independently to each head."""

    def __init__(self, irreps, num_heads, eps=1e-8):
        super().__init__()
        self.num_heads = num_heads
        self.irreps = o3.Irreps(irreps).simplify()
        self.per_head_irrep = _per_head_irrep(self.irreps, num_heads)
        self.norm = EquivariantNorm(self.per_head_irrep, eps=eps)

    def forward(self, x):
        if x.dim() != 3 or x.shape[1] != self.num_heads:
            raise ValueError(f"Expected [N/E, {self.num_heads}, D_head], got {tuple(x.shape)}")
        n = x.shape[0]
        flat = x.reshape(n * self.num_heads, -1)
        out = self.norm(flat)
        return out.reshape(n, self.num_heads, -1)


class NormGate(nn.Module):
    """
    Norm-based gate for modulating non-scalar features.

    Uses the norm (magnitude) of each irrep to gate the features.
    Preserves equivariance by applying elementwise scaling.
    """

    def __init__(self, irrep):
        super(NormGate, self).__init__()
        self.irrep = o3.Irreps(irrep)
        self.norm = o3.Norm(self.irrep)

        num_mul, num_mul_wo_0 = 0, 0
        for mul, ir in self.irrep:
            num_mul += mul
            if ir.l != 0:
                num_mul_wo_0 += mul

        first_mul, first_ir = self.irrep[0]
        if first_ir.l != 0:
            raise ValueError("NormGate expects scalar irreps first")
        self.scalar_mul = first_mul

        if num_mul_wo_0 > 0:
            self.mul = o3.ElementwiseTensorProduct(
                self.irrep[1:], o3.Irreps(f"{num_mul_wo_0}x0e"))
        else:
            self.mul = None
        self.fc = nn.Sequential(
            nn.Linear(num_mul, num_mul),
            nn.SiLU(),
            nn.Linear(num_mul, num_mul))

        self.num_mul = num_mul
        self.num_mul_wo_0 = num_mul_wo_0
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.fc:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        """
        Apply norm-based gating.

        Args:
            x: [N, D] equivariant features

        Returns:
            gated_x: [N, D] gated features
        """
        scalar_slice = self.irrep.slices()[0]
        scalar_x = x[:, scalar_slice]
        norm_x = self.norm(x)[:, self.scalar_mul:]
        f0 = torch.cat([scalar_x, norm_x], dim=-1)
        residual = self.fc(f0)
        scalar_out = scalar_x + residual[:, :self.scalar_mul]
        if self.mul is not None and norm_x.shape[1] > 0:
            gates = 1.0 + residual[:, self.scalar_mul:]
            gated = self.mul(x[:, scalar_slice.stop:], gates)
            x = torch.cat([scalar_out, gated], dim=-1)
        else:
            x = scalar_out
        return x


class ExponentialBernsteinRBF(nn.Module):
    """Exponential Bernstein radial basis functions for distance encoding"""

    def __init__(self, num_basis, max_radius):
        super().__init__()
        self.num_basis = num_basis
        self.max_radius = max_radius
        self.means = nn.Parameter(torch.linspace(0, 1, num_basis))
        self.std = nn.Parameter(torch.ones(1))

    def forward(self, r):
        """
        Args:
            r: [E] or [E, 1] distances

        Returns:
            basis: [E, num_basis] radial basis features
        """
        r = r.squeeze(-1) if r.dim() > 1 else r
        r_norm = r / self.max_radius

        # Gaussian-like basis with learnable centers
        basis = torch.zeros(r.shape[0], self.num_basis, device=r.device, dtype=r.dtype)
        for i in range(self.num_basis):
            basis[:, i] = torch.exp(-((r_norm - self.means[i]) ** 2) / (2 * self.std ** 2 + 1e-8))

        return basis



class MultiHeadAttentionLayer(nn.Module):
    """Multi-head full-irrep inner-product attention."""

    def __init__(
        self,
        irrep_in_node,
        irrep_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        invariant_layers=1,
        invariant_neurons=32,
        nonlinear='ssp',
        use_norm_gate=True,
        attention_temperature=1.0,
        num_heads=4,
        attention_score_residual_init_std=0.0,
    ):
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.use_norm_gate = use_norm_gate
        self.temperature = attention_temperature
        self.num_heads = num_heads
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        self.attention_score_residual_init_std = attention_score_residual_init_std

        self.irrep_in_node = o3.Irreps(irrep_in_node) if not isinstance(irrep_in_node, o3.Irreps) else irrep_in_node
        self.irrep_hidden = o3.Irreps(irrep_hidden) if not isinstance(irrep_hidden, o3.Irreps) else irrep_hidden
        self.irrep_out = o3.Irreps(irrep_out) if not isinstance(irrep_out, o3.Irreps) else irrep_out
        self.sh_irrep = o3.Irreps(sh_irrep) if not isinstance(sh_irrep, o3.Irreps) else sh_irrep

        # Validate multiplicity-axis splitting up front.
        _per_head_irrep(self.irrep_in_node, num_heads)
        _per_head_irrep(self.irrep_hidden, num_heads)

        if nonlinear == 'ssp':
            self.nonlinear = lambda x: torch.nn.functional.softplus(x) - math.log(2.0)
        elif nonlinear == 'silu':
            self.nonlinear = torch.nn.functional.silu
        else:
            self.nonlinear = torch.nn.functional.silu

        self.linear_query = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        self.irrep_tp_key, instruction_key = get_feasible_irrep(
            self.irrep_in_node, self.sh_irrep, self.irrep_in_node, tp_mode='uvu'
        )
        self.tp_key = TensorProduct(
            self.irrep_in_node,
            self.sh_irrep,
            self.irrep_tp_key,
            instruction_key,
            shared_weights=False,
            internal_weights=False,
        )
        self.fc_key = FullyConnectedNet(
            [edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_key.weight_numel],
            self.nonlinear,
        )

        self.irrep_tp_value, instruction_value = get_feasible_irrep(
            self.irrep_in_node, self.sh_irrep, self.irrep_hidden, tp_mode='uvu'
        )
        self.tp_value = TensorProduct(
            self.irrep_in_node,
            self.sh_irrep,
            self.irrep_tp_value,
            instruction_value,
            shared_weights=False,
            internal_weights=False,
        )
        self.fc_value = FullyConnectedNet(
            [edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_value.weight_numel],
            self.nonlinear,
        )

        self.inner_product = MultiHeadInnerProduct(self.irrep_in_node, num_heads)
        self.attention_score = InvariantAttentionScore(
            self.inner_product.irrep_out.dim,
            num_heads,
            hidden_dim=invariant_neurons,
            residual_init_std=attention_score_residual_init_std,
        )
        self.query_norm = MultiHeadEquivariantNorm(self.irrep_in_node, num_heads)
        self.key_norm = MultiHeadEquivariantNorm(self.irrep_tp_key, num_heads)

        num_mul = sum(mul for mul, _ in self.irrep_in_node)
        self.inner_product_s0 = InnerProduct(self.irrep_in_node)
        s0_dim = num_mul + self.irrep_in_node[0][0]
        self.layer_l0_key = FullyConnectedNet(
            [s0_dim] + invariant_layers * [invariant_neurons] + [self.tp_key.weight_numel],
            self.nonlinear,
        )
        self.layer_l0_value = FullyConnectedNet(
            [s0_dim] + invariant_layers * [invariant_neurons] + [self.tp_value.weight_numel],
            self.nonlinear,
        )

        self.linear_out = Linear(
            irreps_in=self.irrep_hidden,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True,
        )

        if use_norm_gate:
            self.norm_gate = NormGate(self.irrep_in_node)
            self.irrep_linear_out, _ = get_feasible_irrep(
                self.irrep_in_node, o3.Irreps("0e"), self.irrep_in_node
            )
            self.linear_node = Linear(
                irreps_in=self.irrep_in_node,
                irreps_out=self.irrep_linear_out,
                internal_weights=True,
                shared_weights=True,
                biases=True,
            )
            self.linear_node_pre = Linear(
                irreps_in=self.irrep_in_node,
                irreps_out=self.irrep_linear_out,
                internal_weights=True,
                shared_weights=True,
                biases=True,
            )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, FullyConnectedNet):
                for sub_module in module:
                    if isinstance(sub_module, nn.Linear):
                        nn.init.normal_(sub_module.weight, mean=0.0, std=0.01)
                        if sub_module.bias is not None:
                            nn.init.zeros_(sub_module.bias)
        if hasattr(self, "attention_score"):
            self.attention_score.reset_parameters()

    def _compute_s0(self, x, edge_dst, edge_src):
        if self.use_norm_gate:
            pre_x = self.linear_node_pre(x)
            s0 = self.inner_product_s0(pre_x[edge_dst], pre_x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat([
                pre_x[edge_dst][:, self.irrep_in_node.slices()[0]],
                pre_x[edge_src][:, self.irrep_in_node.slices()[0]],
                s0,
            ], dim=-1)
        else:
            s0 = self.inner_product_s0(x[edge_dst], x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat([
                x[edge_dst][:, self.irrep_in_node.slices()[0]],
                x[edge_src][:, self.irrep_in_node.slices()[0]],
                s0,
            ], dim=-1)
        return s0

    def _project_key_value(self, x, edge_src, edge_sh, edge_attr, s0):
        key = self.tp_key(
            x[edge_src],
            edge_sh,
            self.fc_key(edge_attr) * self.layer_l0_key(s0),
        )
        value = self.tp_value(
            x[edge_src],
            edge_sh,
            self.fc_value(edge_attr) * self.layer_l0_value(s0),
        )
        return key, value

    def _multihead_attention(self, query, key, value, edge_dst, key_irrep, value_irrep, num_nodes):
        query_heads = split_irreps_multiplicity(query, key_irrep, self.num_heads)
        key_heads = split_irreps_multiplicity(key, key_irrep, self.num_heads)
        value_heads = split_irreps_multiplicity(value, value_irrep, self.num_heads)

        query_norm = self.query_norm(query_heads)
        key_norm = self.key_norm(key_heads)

        attention_input = self.inner_product(query_norm[edge_dst], key_norm)
        attention_input = attention_input.clamp(max=5.0, min=-5.0)
        attention_logits = self.attention_score(attention_input) / self.temperature
        attention_logits = attention_logits.clamp(max=10.0, min=-10.0)

        attention_stable = attention_logits - attention_logits.max(dim=0, keepdim=True)[0]
        exp_logits = attention_stable.exp()
        z = scatter(exp_logits, edge_dst, dim=0, dim_size=num_nodes).clamp(min=1e-10)
        alpha = (exp_logits / z[edge_dst]).clamp(min=0.0, max=1.0)

        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            alpha = torch.ones_like(alpha) / scatter(
                torch.ones_like(alpha),
                edge_dst,
                dim=0,
                dim_size=num_nodes,
                reduce='sum',
            )[edge_dst].clamp(min=1.0)

        attended_heads = scatter(
            alpha.unsqueeze(-1) * value_heads,
            edge_dst,
            dim=0,
            dim_size=num_nodes,
        )
        return merge_heads(attended_heads, value_irrep, self.num_heads)

    def forward(self, data, x):
        edge_dst, edge_src = data.edge_index[0], data.edge_index[1]
        num_nodes = x.shape[0]
        self_x = x

        if self.use_norm_gate:
            x = self.norm_gate(x)
            x = self.linear_node(x)

        s0 = self._compute_s0(x, edge_dst, edge_src)
        query = self.linear_query(x)
        key, value = self._project_key_value(x, edge_src, data.edge_sh, data.edge_attr, s0)
        attended = self._multihead_attention(
            query, key, value, edge_dst,
            self.irrep_tp_key, self.irrep_tp_value, num_nodes,
        )

        out = self.linear_out(attended)
        if self.irrep_in_node == self.irrep_out:
            out = out + self_x
        return out


class MultiHeadAttentionNetLayer(nn.Module):
    """Wrapper with the same interface as the previous GNN attention wrapper."""

    def __init__(
        self,
        irrep_in_node,
        irrep_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        use_norm_gate=True,
        attention_temperature=1.0,
        num_heads=4,
        attention_score_residual_init_std=0.0,
    ):
        super().__init__()
        self.irrep_in_node = o3.Irreps(irrep_in_node) if not isinstance(irrep_in_node, o3.Irreps) else irrep_in_node
        self.irrep_out = o3.Irreps(irrep_out) if not isinstance(irrep_out, o3.Irreps) else irrep_out
        self.resnet = resnet and self.irrep_in_node == self.irrep_out
        self.conv = MultiHeadAttentionLayer(
            irrep_in_node=irrep_in_node,
            irrep_hidden=irrep_hidden,
            irrep_out=irrep_out,
            sh_irrep=sh_irrep,
            edge_attr_dim=edge_attr_dim,
            node_attr_dim=node_attr_dim,
            invariant_layers=1,
            invariant_neurons=32,
            nonlinear='ssp',
            use_norm_gate=use_norm_gate,
            attention_temperature=attention_temperature,
            num_heads=num_heads,
            attention_score_residual_init_std=attention_score_residual_init_std,
        )

    def forward(self, data, x):
        return self.conv(data, x)


class CompressedSparseAttentionLayer(MultiHeadAttentionLayer):
    """Multi-head CSA with invariant Lightning Indexer top-k pruning."""

    def __init__(self, *args, top_k=8, indexer_compress_dim=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = top_k
        self.indexer_compress_dim = indexer_compress_dim

        num_mul = sum(mul for mul, _ in self.irrep_in_node)
        s0_dim = num_mul + self.irrep_in_node[0][0]
        scorer_in_dim = s0_dim + self.edge_attr_dim
        self.indexer = nn.Sequential(
            nn.Linear(scorer_in_dim, indexer_compress_dim),
            nn.SiLU(),
            nn.Linear(indexer_compress_dim, 1),
        )
        for module in self.indexer:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
                nn.init.zeros_(module.bias)

    def _select_topk(self, relevance, edge_dst, top_k):
        if top_k is None or edge_dst.numel() == 0:
            return torch.arange(edge_dst.numel(), dtype=torch.long, device=edge_dst.device)

        scores = relevance.squeeze(-1)
        edge_order = torch.argsort(scores, descending=True, stable=True)
        dst_order = torch.argsort(edge_dst[edge_order], stable=True)
        ordered = edge_order[dst_order]
        ordered_dst = edge_dst[ordered]

        positions = torch.arange(ordered.numel(), device=edge_dst.device)
        is_group_start = torch.ones_like(ordered_dst, dtype=torch.bool)
        is_group_start[1:] = ordered_dst[1:] != ordered_dst[:-1]
        group_starts = torch.where(is_group_start, positions, torch.zeros_like(positions))
        group_starts = torch.cummax(group_starts, dim=0).values
        rank_in_group = positions - group_starts
        return ordered[rank_in_group < top_k]

    def forward(self, data, x):
        edge_dst, edge_src = data.edge_index[0], data.edge_index[1]
        num_nodes = x.shape[0]
        self_x = x

        if self.use_norm_gate:
            x = self.norm_gate(x)
            x = self.linear_node(x)

        s0_full = self._compute_s0(x, edge_dst, edge_src)
        relevance = self.indexer(torch.cat([s0_full, data.edge_attr], dim=-1))
        sel_mask = self._select_topk(relevance, edge_dst, self.top_k)
        sel_dst = edge_dst[sel_mask]
        sel_src = edge_src[sel_mask]

        query = self.linear_query(x)
        key, value = self._project_key_value(
            x,
            sel_src,
            data.edge_sh[sel_mask],
            data.edge_attr[sel_mask],
            s0_full[sel_mask],
        )
        attended = self._multihead_attention(
            query, key, value, sel_dst,
            self.irrep_tp_key, self.irrep_tp_value, num_nodes,
        )

        out = self.linear_out(attended)
        if self.irrep_in_node == self.irrep_out:
            out = out + self_x
        return out


class CompressedSparseAttentionNetLayer(nn.Module):
    """Wrapper for ``CompressedSparseAttentionLayer``."""

    def __init__(
        self,
        irrep_in_node,
        irrep_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        use_norm_gate=True,
        attention_temperature=1.0,
        num_heads=4,
        top_k=8,
        indexer_compress_dim=32,
        attention_score_residual_init_std=0.0,
    ):
        super().__init__()
        self.irrep_in_node = o3.Irreps(irrep_in_node) if not isinstance(irrep_in_node, o3.Irreps) else irrep_in_node
        self.irrep_out = o3.Irreps(irrep_out) if not isinstance(irrep_out, o3.Irreps) else irrep_out
        self.resnet = resnet and self.irrep_in_node == self.irrep_out
        self.conv = CompressedSparseAttentionLayer(
            irrep_in_node=irrep_in_node,
            irrep_hidden=irrep_hidden,
            irrep_out=irrep_out,
            sh_irrep=sh_irrep,
            edge_attr_dim=edge_attr_dim,
            node_attr_dim=node_attr_dim,
            invariant_layers=1,
            invariant_neurons=32,
            nonlinear='ssp',
            use_norm_gate=use_norm_gate,
            attention_temperature=attention_temperature,
            num_heads=num_heads,
            top_k=top_k,
            indexer_compress_dim=indexer_compress_dim,
            attention_score_residual_init_std=attention_score_residual_init_std,
        )

    def forward(self, data, x):
        return self.conv(data, x)


class HeavyCompressedAttentionLayer(MultiHeadAttentionLayer):
    """Multi-head HCA with l<=hca_lmax K/V compression and zero padding."""

    def __init__(self, *args, hca_lmax=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.hca_lmax = hca_lmax
        self.hca_key_irrep = _filter_irrep_lmax(self.irrep_in_node, hca_lmax)
        self.hca_value_irrep = _filter_irrep_lmax(self.irrep_hidden, hca_lmax)

        self.irrep_tp_key, instruction_key = get_feasible_irrep(
            self.irrep_in_node, self.sh_irrep, self.hca_key_irrep, tp_mode='uvu'
        )
        self.tp_key = TensorProduct(
            self.irrep_in_node,
            self.sh_irrep,
            self.irrep_tp_key,
            instruction_key,
            shared_weights=False,
            internal_weights=False,
        )
        self.fc_key = FullyConnectedNet(
            [self.edge_attr_dim] + self.invariant_layers * [self.invariant_neurons] + [self.tp_key.weight_numel],
            self.nonlinear,
        )

        self.irrep_tp_value, instruction_value = get_feasible_irrep(
            self.irrep_in_node, self.sh_irrep, self.hca_value_irrep, tp_mode='uvu'
        )
        self.tp_value = TensorProduct(
            self.irrep_in_node,
            self.sh_irrep,
            self.irrep_tp_value,
            instruction_value,
            shared_weights=False,
            internal_weights=False,
        )
        self.fc_value = FullyConnectedNet(
            [self.edge_attr_dim] + self.invariant_layers * [self.invariant_neurons] + [self.tp_value.weight_numel],
            self.nonlinear,
        )

        num_mul = sum(mul for mul, _ in self.irrep_in_node)
        s0_dim = num_mul + self.irrep_in_node[0][0]
        self.layer_l0_key = FullyConnectedNet(
            [s0_dim] + self.invariant_layers * [self.invariant_neurons] + [self.tp_key.weight_numel],
            self.nonlinear,
        )
        self.layer_l0_value = FullyConnectedNet(
            [s0_dim] + self.invariant_layers * [self.invariant_neurons] + [self.tp_value.weight_numel],
            self.nonlinear,
        )
        self.inner_product = MultiHeadInnerProduct(self.hca_key_irrep, self.num_heads)
        self.attention_score = InvariantAttentionScore(
            self.inner_product.irrep_out.dim,
            self.num_heads,
            hidden_dim=self.invariant_neurons,
            residual_init_std=self.attention_score_residual_init_std,
        )
        self.query_norm = MultiHeadEquivariantNorm(self.hca_key_irrep, self.num_heads)
        self.key_norm = MultiHeadEquivariantNorm(self.irrep_tp_key, self.num_heads)
        self._init_weights()

    def forward(self, data, x):
        edge_dst, edge_src = data.edge_index[0], data.edge_index[1]
        num_nodes = x.shape[0]
        self_x = x

        if self.use_norm_gate:
            x = self.norm_gate(x)
            x = self.linear_node(x)

        s0 = self._compute_s0(x, edge_dst, edge_src)
        query = self.linear_query(x)
        query_hca = filter_irrep_tensor_lmax(query, self.irrep_in_node, self.hca_lmax)
        key, value = self._project_key_value(x, edge_src, data.edge_sh, data.edge_attr, s0)
        attended_hca = self._multihead_attention(
            query_hca, key, value, edge_dst,
            self.irrep_tp_key, self.irrep_tp_value, num_nodes,
        )
        attended = pad_irrep_tensor(attended_hca, self.irrep_tp_value, self.irrep_hidden)

        out = self.linear_out(attended)
        if self.irrep_in_node == self.irrep_out:
            out = out + self_x
        return out


class HeavyCompressedAttentionNetLayer(nn.Module):
    """Wrapper for ``HeavyCompressedAttentionLayer``."""

    def __init__(
        self,
        irrep_in_node,
        irrep_hidden,
        irrep_out,
        sh_irrep,
        edge_attr_dim,
        node_attr_dim,
        resnet: bool = True,
        use_norm_gate=True,
        attention_temperature=1.0,
        num_heads=4,
        hca_lmax=2,
        attention_score_residual_init_std=0.0,
    ):
        super().__init__()
        self.irrep_in_node = o3.Irreps(irrep_in_node) if not isinstance(irrep_in_node, o3.Irreps) else irrep_in_node
        self.irrep_out = o3.Irreps(irrep_out) if not isinstance(irrep_out, o3.Irreps) else irrep_out
        self.resnet = resnet and self.irrep_in_node == self.irrep_out
        self.conv = HeavyCompressedAttentionLayer(
            irrep_in_node=irrep_in_node,
            irrep_hidden=irrep_hidden,
            irrep_out=irrep_out,
            sh_irrep=sh_irrep,
            edge_attr_dim=edge_attr_dim,
            node_attr_dim=node_attr_dim,
            invariant_layers=1,
            invariant_neurons=32,
            nonlinear='ssp',
            use_norm_gate=use_norm_gate,
            attention_temperature=attention_temperature,
            num_heads=num_heads,
            hca_lmax=hca_lmax,
            attention_score_residual_init_std=attention_score_residual_init_std,
        )

    def forward(self, data, x):
        return self.conv(data, x)


if __name__ == "__main__":
    # Test the multi-head inner-product attention layer
    print("Testing MultiHeadAttentionLayer...")

    # Create dummy data
    num_nodes = 10
    num_edges = 30
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Irreps
    irrep_node = o3.Irreps("128x0e + 128x1o + 128x2e")
    irrep_hidden = o3.Irreps("128x0e + 128x1o + 128x2e")
    sh_irrep = o3.Irreps.spherical_harmonics(lmax=4)

    # Dummy features
    x = torch.randn(num_nodes, irrep_node.dim)
    edge_index = torch.stack([
        torch.randint(0, num_nodes, (num_edges,)),
        torch.randint(0, num_nodes, (num_edges,))
    ])
    edge_attr = torch.randn(num_edges, 32)
    edge_sh = torch.randn(num_edges, sh_irrep.dim)

    # Create data object
    class Data:
        pass
    data = Data()
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    data.edge_sh = edge_sh

    # Create layer
    layer = MultiHeadAttentionLayer(
        irrep_in_node=irrep_node,
        irrep_hidden=irrep_hidden,
        irrep_out=irrep_node,
        sh_irrep=sh_irrep,
        edge_attr_dim=32,
        node_attr_dim=128,
        use_norm_gate=True,
        attention_temperature=1.0,
        num_heads=4,
    )

    # Forward pass
    out = layer(data, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"MultiHeadAttentionLayer test passed!")
