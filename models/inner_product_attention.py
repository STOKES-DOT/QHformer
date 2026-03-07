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

# Scatter function with fallback
try:
    from torch_scatter import scatter as torch_scatter_fn
    HAS_TORCH_SCATTER = True
except ImportError:
    HAS_TORCH_SCATTER = False


def scatter_pure(src, index, dim_size=None, dim=0, reduce='sum'):
    """Pure PyTorch scatter implementation"""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    if reduce == 'sum' or reduce == 'add':
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src) if index.dim() == 1 else index, src)
    elif reduce == 'mean':
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        counts = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src) if index.dim() == 1 else index, src)
        counts.scatter_add_(dim, index.unsqueeze(-1).expand_as(src) if index.dim() == 1 else index, torch.ones_like(src))
        return out / (counts.clamp(min=1))
    else:
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src) if index.dim() == 1 else index, src)


def scatter(src, index, dim_size=None, dim=0, reduce='sum'):
    """Scatter function with fallback"""
    if HAS_TORCH_SCATTER:
        return torch_scatter_fn(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
    else:
        return scatter_pure(src, index, dim_size=dim_size, dim=dim, reduce=reduce)


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

    def forward(self, x):
        """
        Apply norm-based gating.

        Args:
            x: [N, D] equivariant features

        Returns:
            gated_x: [N, D] gated features
        """
        norm_x = self.norm(x)[:, self.irrep.slices()[0].stop:]
        f0 = torch.cat([x[:, self.irrep.slices()[0]], norm_x], dim=-1)
        gates = self.fc(f0)
        if self.mul is not None and norm_x.shape[1] > 0:
            gated = self.mul(x[:, self.irrep.slices()[0].stop:], gates[:, self.irrep.slices()[0].stop:])
            x = torch.cat([gates[:, self.irrep.slices()[0]], gated], dim=-1)
        else:
            x = gates[:, self.irrep.slices()[0]]
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


class InnerProductAttentionLayer(nn.Module):
    """
    Scheme 1: Inner Product Attention Layer

    This layer uses InnerProduct to couple Query and Key irreps to scalars
    for computing attention weights. The Value maintains full irreps.

    Architecture:
        1. Query: Linear(node_i) → full_irreps
        2. Key: TP(node_j, edge_sh, W_k) → full_irreps
        3. Value: TP(node_j, edge_sh, W_v) → hidden_irreps
        4. Attention: α_ij = softmax(IP(Query_i, Key_j) / √d)
        5. Output: Σ_j α_ij · Value_j

    Key Features:
    - Query and Key maintain full irreps (no scalar compression)
    - InnerProduct couples irreps to rotation-invariant scalars
    - Value preserves full equivariant information
    - Fully SO(3) equivariant

    Args:
        irrep_in_node: Input node irreps
        irrep_hidden: Hidden irreps for value
        irrep_out: Output irreps
        sh_irrep: Spherical harmonics irreps for edges
        edge_attr_dim: Edge attribute dimension
        node_attr_dim: Node attribute dimension
        invariant_layers: Number of invariant layers for weight networks
        invariant_neurons: Number of neurons in invariant layers
        nonlinear: Nonlinear activation ('ssp', 'silu', etc.)
        use_norm_gate: Whether to use norm gate
        attention_temperature: Temperature for attention (1/√d)
    """

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
    ):
        super(InnerProductAttentionLayer, self).__init__()

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.use_norm_gate = use_norm_gate
        self.temperature = attention_temperature

        # Convert to Irreps
        self.irrep_in_node = o3.Irreps(irrep_in_node) if not isinstance(irrep_in_node, o3.Irreps) else irrep_in_node
        self.irrep_hidden = o3.Irreps(irrep_hidden) if not isinstance(irrep_hidden, o3.Irreps) else irrep_hidden
        self.irrep_out = o3.Irreps(irrep_out) if not isinstance(irrep_out, o3.Irreps) else irrep_out
        self.sh_irrep = o3.Irreps(sh_irrep) if not isinstance(sh_irrep, o3.Irreps) else sh_irrep

        # Nonlinear activation
        if nonlinear == 'ssp':
            self.nonlinear = lambda x: torch.nn.functional.softplus(x) - math.log(2.0)
        elif nonlinear == 'silu':
            self.nonlinear = torch.nn.functional.silu
        else:
            self.nonlinear = torch.nn.functional.silu

        # ============ Query Projection ============
        # Query: Keep FULL irreps (no scalar compression!)
        self.linear_query = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        # ============ Key TensorProduct ============
        # Key: node ⊗ edge_sh → same_irreps (keep full irreps)
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
            self.nonlinear
        )

        # ============ Value TensorProduct ============
        # Value: node ⊗ edge_sh → hidden_irreps
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
            self.nonlinear
        )

        # ============ InnerProduct for Attention ============
        # InnerProduct: Query ⊗ Key → scalar (rotation invariant)
        self.inner_product = InnerProduct(self.irrep_in_node)

        # Equivariant normalization for numerical stability
        self.query_norm = EquivariantNorm(self.irrep_in_node)
        self.key_norm = EquivariantNorm(self.irrep_tp_key)

        # ============ Additional Components ============
        # Inner product for s0 (node similarity modulation)
        num_mul = sum(mul for mul, _ in self.irrep_in_node)
        self.inner_product_s0 = InnerProduct(self.irrep_in_node)

        # Layer for s0 modulation
        self.layer_l0_key = FullyConnectedNet(
            [num_mul + self.irrep_in_node[0][0]] + invariant_layers * [invariant_neurons] + [self.tp_key.weight_numel],
            self.nonlinear
        )

        self.layer_l0_value = FullyConnectedNet(
            [num_mul + self.irrep_in_node[0][0]] + invariant_layers * [invariant_neurons] + [self.tp_value.weight_numel],
            self.nonlinear
        )

        # ============ Output Projection ============
        self.linear_out = Linear(
            irreps_in=self.irrep_hidden,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        # ============ NormGate (optional) ============
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
                biases=True
            )
            self.linear_node_pre = Linear(
                irreps_in=self.irrep_in_node,
                irreps_out=self.irrep_linear_out,
                internal_weights=True,
                shared_weights=True,
                biases=True
            )

        # Initialize weights for numerical stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for numerical stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, FullyConnectedNet):
                # FullyConnectedNet uses nn.Sequential
                for sub_module in module:
                    if isinstance(sub_module, nn.Linear):
                        nn.init.normal_(sub_module.weight, mean=0.0, std=0.01)
                        if sub_module.bias is not None:
                            nn.init.zeros_(sub_module.bias)

    def _compute_s0(self, x, edge_dst, edge_src):
        """Compute invariant scalar features from node pair for modulation"""
        if self.use_norm_gate:
            pre_x = self.linear_node_pre(x)
            s0 = self.inner_product_s0(pre_x[edge_dst], pre_x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat([
                pre_x[edge_dst][:, self.irrep_in_node.slices()[0]],
                pre_x[edge_src][:, self.irrep_in_node.slices()[0]],
                s0
            ], dim=-1)
        else:
            s0 = self.inner_product_s0(x[edge_dst], x[edge_src])[:, self.irrep_in_node.slices()[0].stop:]
            s0 = torch.cat([
                x[edge_dst][:, self.irrep_in_node.slices()[0]],
                x[edge_src][:, self.irrep_in_node.slices()[0]],
                s0
            ], dim=-1)
        return s0

    def forward(self, data, x):
        """
        Forward pass with Inner Product Attention

        Args:
            data: Graph data with edge_index, edge_attr, edge_sh
            x: Node features [N, D_in]

        Returns:
            out: Updated node features [N, D_out]
        """
        edge_dst, edge_src = data.edge_index[0], data.edge_index[1]

        # Apply NormGate if needed
        if self.use_norm_gate:
            x = self.norm_gate(x)
            x = self.linear_node(x)

        self_x = x  # For residual connection

        # Compute s0 for modulation (shared)
        s0 = self._compute_s0(x, edge_dst, edge_src)

        # ============ Compute Query ============
        # Query: Keep FULL irreps (no compression to scalar!)
        query = self.linear_query(x)  # [N, D_in] with full irreps

        # ============ Compute Key ============
        # Key: TP(node_j, edge_sh, W_k * s0)
        key_hidden = self.tp_key(
            x[edge_src],
            data.edge_sh,
            self.fc_key(data.edge_attr) * self.layer_l0_key(s0)
        )  # [E, D_in] with full irreps

        # ============ Compute Value ============
        # Value: TP(node_j, edge_sh, W_v * s0)
        value_edge = self.tp_value(
            x[edge_src],
            data.edge_sh,
            self.fc_value(data.edge_attr) * self.layer_l0_value(s0)
        )  # [E, D_hidden]

        # ============ Compute Attention Weights via InnerProduct ============
        # InnerProduct couples Query and Key to rotation-invariant scalars
        # IP(Query_i, Key_j) = Σ_l Σ_m Query_i^(l,m) · Key_j^(l,m)

        # Normalize Query and Key BEFORE InnerProduct to prevent numerical overflow
        # This is the KEY fix for NaN issues!
        query_normalized = self.query_norm(query)  # [N, D]
        key_normalized = self.key_norm(key_hidden)  # [E, D]

        # EQUIVARIANCE FIX: REMOVED clamp() on query_normalized and key_normalized
        # Applying element-wise clipping to high-order irreps (vectors, tensors) destroys SO(3) equivariance!
        # Only clamp the scalar attention scores below (which is mathematically safe).

        attention_input = self.inner_product(query_normalized[edge_dst], key_normalized)  # [E, num_scalars]

        # Clamp each scalar channel BEFORE summing to prevent overflow
        attention_input = attention_input.clamp(max=5.0, min=-5.0)  # Tighter clamping

        # Sum over all scalar channels to get final attention score
        attention_logits = attention_input.sum(dim=-1, keepdim=True)  # [E, 1]
        attention_logits = attention_logits / self.temperature

        # More aggressive clamping for numerical stability
        attention_logits = attention_logits.clamp(max=10.0, min=-10.0)

        # Use softmax for better numerical stability
        attention_logits_stable = attention_logits - attention_logits.max(dim=0, keepdim=True)[0]
        exp_logits = attention_logits_stable.exp()
        z = scatter(exp_logits, edge_dst, dim=0, dim_size=len(x))
        z = z.clamp(min=1e-10)  # Avoid division by zero
        alpha = exp_logits / z[edge_dst]

        # Clamp attention weights
        alpha = alpha.clamp(min=0.0, max=1.0)

        # Check for NaN/Inf - fall back to uniform attention
        if torch.isnan(alpha).any() or torch.isinf(alpha).any():
            alpha = torch.ones_like(alpha) / scatter(
                torch.ones_like(alpha), edge_dst, dim=0, dim_size=len(x), reduce='sum'
            )[edge_dst]

        # ============ Aggregate with Attention Weights ============
        # Use simple weighted sum for stability (instead of sqrt(alpha))
        attended_features = scatter(
            (alpha * value_edge),
            edge_dst,
            dim=0,
            dim_size=len(x)
        )

        # ============ Output Projection ============
        out = self.linear_out(attended_features)

        # EQUIVARIANCE FIX: REMOVED clamp() on output tensor
        # Clamping high-order irreps destroys rotational equivariance!
        # The normalization layers should handle numerical stability.

        # ============ Residual Connection ============
        if self.irrep_in_node == self.irrep_out:
            out = out + self_x

        return out


class InnerProductAttentionNetLayer(nn.Module):
    """Wrapper with same interface as standard ConvLayer"""

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
    ):
        super(InnerProductAttentionNetLayer, self).__init__()

        self.irrep_in_node = o3.Irreps(irrep_in_node) if not isinstance(irrep_in_node, o3.Irreps) else irrep_in_node
        self.irrep_out = o3.Irreps(irrep_out) if not isinstance(irrep_out, o3.Irreps) else irrep_out
        self.resnet = resnet and self.irrep_in_node == self.irrep_out

        self.conv = InnerProductAttentionLayer(
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
        )

    def forward(self, data, x):
        out = self.conv(data, x)
        return out


if __name__ == "__main__":
    # Test the Inner Product Attention layer
    print("Testing InnerProductAttentionLayer...")

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
    layer = InnerProductAttentionLayer(
        irrep_in_node=irrep_node,
        irrep_hidden=irrep_hidden,
        irrep_out=irrep_node,
        sh_irrep=sh_irrep,
        edge_attr_dim=32,
        node_attr_dim=128,
        use_norm_gate=True,
        attention_temperature=1.0,
    )

    # Forward pass
    out = layer(data, x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"InnerProductAttentionLayer test passed!")
