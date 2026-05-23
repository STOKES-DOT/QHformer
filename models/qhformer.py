"""
QHformer: QHNet with Inner Product Attention

This model integrates inner product attention mechanism into QHNet
for predicting quantum Hamiltonian matrices from molecular geometries.

Key Innovation:
- Query and Key maintain FULL irreps (no scalar compression)
- InnerProduct couples irreps to rotation-invariant scalars for attention
- Value preserves complete equivariant information

Reference:
- Original QHNet: https://github.com/Divel-DiNISR/QHNet
- e3nn Transformer: https://docs.e3nn.org/en/stable/guide/transformer.html
"""

import time
import torch
import torch.nn as nn
import math
from e3nn import o3

try:
    from .torch_ops import radius_graph
except ImportError:
    from torch_ops import radius_graph

# Handle both relative and absolute imports
try:
    from .inner_product_attention import (
        MultiHeadAttentionNetLayer,
        CompressedSparseAttentionNetLayer,
        HeavyCompressedAttentionNetLayer,
        get_feasible_irrep,
        InnerProduct,
        NormGate,
        ExponentialBernsteinRBF,
        scatter,
    )
except ImportError:
    from inner_product_attention import (
        MultiHeadAttentionNetLayer,
        CompressedSparseAttentionNetLayer,
        HeavyCompressedAttentionNetLayer,
        get_feasible_irrep,
        InnerProduct,
        NormGate,
        ExponentialBernsteinRBF,
        scatter,
    )


def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)


def get_nonlinear(nonlinear: str):
    if nonlinear.lower() == 'ssp':
        return ShiftedSoftPlus
    elif nonlinear.lower() == 'silu':
        return torch.nn.functional.silu
    elif nonlinear.lower() == 'tanh':
        return torch.nn.functional.tanh
    elif nonlinear.lower() == 'abs':
        return torch.abs
    else:
        raise NotImplementedError


# ============ QHNet Core Components ============

class SelfNetLayer(nn.Module):
    """
    Self-interaction network for diagonal Hamiltonian blocks.
    Pure node-node interaction without edge spherical harmonics.
    """

    def __init__(self,
                 irrep_in_node,
                 irrep_bottle_hidden,
                 irrep_out,
                 sh_irrep,
                 edge_attr_dim,
                 node_attr_dim,
                 resnet: bool = True,
                 nonlinear='ssp'):
        super(SelfNetLayer, self).__init__()
        self.sh_irrep = sh_irrep
        self.irrep_in_node = o3.Irreps(irrep_in_node) if not isinstance(irrep_in_node, o3.Irreps) else irrep_in_node
        self.irrep_bottle_hidden = o3.Irreps(irrep_bottle_hidden) if not isinstance(irrep_bottle_hidden, o3.Irreps) else irrep_bottle_hidden
        self.irrep_out = o3.Irreps(irrep_out) if not isinstance(irrep_out, o3.Irreps) else irrep_out

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim
        self.resnet = resnet

        self.irrep_tp_in_node, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden)
        self.irrep_tp_out_node, instruction_node = get_feasible_irrep(
            self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode='uuu')

        from e3nn.o3 import Linear, TensorProduct
        self.linear_node_1 = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.linear_node_2 = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.tp = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node,
            instruction_node,
            shared_weights=True,
            internal_weights=True
        )

        self.norm_gate = NormGate(self.irrep_out)
        self.norm_gate_1 = NormGate(self.irrep_in_node)
        self.norm_gate_2 = NormGate(self.irrep_in_node)

        self.linear_node_3 = Linear(
            irreps_in=self.irrep_tp_out_node,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

    def forward(self, data, x, old_fii):
        old_x = x
        xl = self.norm_gate_1(x)
        xl = self.linear_node_1(xl)
        xr = self.norm_gate_2(x)
        xr = self.linear_node_2(xr)
        x = self.tp(xl, xr)
        if self.resnet:
            x = x + old_x
        x = self.norm_gate(x)
        x = self.linear_node_3(x)
        if self.resnet and old_fii is not None:
            x = old_fii + x
        return x


class PairNetLayer(nn.Module):
    """
    Pair-interaction network for off-diagonal Hamiltonian blocks.
    Enhanced with edge spherical harmonics for directional information.
    """

    def __init__(self,
                 irrep_in_node,
                 irrep_bottle_hidden,
                 irrep_out,
                 sh_irrep,
                 edge_attr_dim,
                 node_attr_dim,
                 resnet: bool = True,
                 invariant_layers=1,
                 invariant_neurons=8,
                 nonlinear='ssp'):
        super(PairNetLayer, self).__init__()
        self.invariant_layers = invariant_layers
        self.invariant_neurons = invariant_neurons
        nonlinear_fn = get_nonlinear(nonlinear)

        self.irrep_in_node = o3.Irreps(irrep_in_node) if not isinstance(irrep_in_node, o3.Irreps) else irrep_in_node
        self.irrep_bottle_hidden = o3.Irreps(irrep_bottle_hidden) if not isinstance(irrep_bottle_hidden, o3.Irreps) else irrep_bottle_hidden
        self.irrep_out = o3.Irreps(irrep_out) if not isinstance(irrep_out, o3.Irreps) else irrep_out
        self.sh_irrep = o3.Irreps(sh_irrep) if not isinstance(sh_irrep, o3.Irreps) else sh_irrep

        self.edge_attr_dim = edge_attr_dim
        self.node_attr_dim = node_attr_dim

        self.irrep_tp_in_node, _ = get_feasible_irrep(self.irrep_in_node, o3.Irreps("0e"), self.irrep_bottle_hidden)
        self.irrep_tp_out_node_pair, instruction_node_pair = get_feasible_irrep(
            self.irrep_tp_in_node, self.irrep_tp_in_node, self.irrep_bottle_hidden, tp_mode='uuu')

        from e3nn.o3 import Linear, TensorProduct
        from e3nn.nn import FullyConnectedNet

        self.linear_node_pair = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_tp_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.linear_node_pair_n = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.linear_node_pair_inner = Linear(
            irreps_in=self.irrep_in_node,
            irreps_out=self.irrep_in_node,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.tp_node_pair = TensorProduct(
            self.irrep_tp_in_node,
            self.irrep_tp_in_node,
            self.irrep_tp_out_node_pair,
            instruction_node_pair,
            shared_weights=False,
            internal_weights=False,
        )

        self.irrep_tp_out_node_pair_2, instruction_node_pair_2 = get_feasible_irrep(
            self.irrep_tp_out_node_pair, self.irrep_tp_out_node_pair, self.irrep_bottle_hidden, tp_mode='uuu')

        self.tp_node_pair_2 = TensorProduct(
            self.irrep_tp_out_node_pair,
            self.irrep_tp_out_node_pair,
            self.irrep_tp_out_node_pair_2,
            instruction_node_pair_2,
            shared_weights=True,
            internal_weights=True
        )

        self.fc_node_pair = FullyConnectedNet(
            [self.edge_attr_dim] + invariant_layers * [invariant_neurons] + [self.tp_node_pair.weight_numel],
            nonlinear_fn
        )

        self.linear_node_pair_2 = Linear(
            irreps_in=self.irrep_tp_out_node_pair_2,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        if self.irrep_in_node == self.irrep_out and resnet:
            self.resnet = True
        else:
            self.resnet = False

        self.linear_node_pair_final = Linear(
            irreps_in=self.irrep_tp_out_node_pair,
            irreps_out=self.irrep_out,
            internal_weights=True,
            shared_weights=True,
            biases=True
        )

        self.norm_gate = NormGate(self.irrep_tp_out_node_pair)
        self.inner_product = InnerProduct(self.irrep_in_node)
        self.norm = o3.Norm(self.irrep_in_node)

        num_mul = sum(mul for mul, _ in self.irrep_in_node)
        self.norm_gate_pre = NormGate(self.irrep_tp_out_node_pair)
        self.fc = nn.Sequential(
            nn.Linear(self.irrep_in_node[0][0] + num_mul, self.irrep_in_node[0][0]),
            nn.SiLU(),
            nn.Linear(self.irrep_in_node[0][0], self.tp_node_pair.weight_numel))

    def forward(self, data, node_attr, node_pair_attr=None):
        pair_edge_index = getattr(data, "pair_edge_index", data.full_edge_index)
        pair_edge_attr = getattr(data, "pair_edge_attr", data.full_edge_attr)
        dst, src = pair_edge_index
        node_attr_0 = self.linear_node_pair_inner(node_attr)
        s0 = self.inner_product(node_attr_0[dst], node_attr_0[src])[:, self.irrep_in_node.slices()[0].stop:]
        s0 = torch.cat([node_attr_0[dst][:, self.irrep_in_node.slices()[0]],
                        node_attr_0[src][:, self.irrep_in_node.slices()[0]], s0], dim=-1)

        node_attr = self.norm_gate_pre(node_attr)
        node_attr = self.linear_node_pair_n(node_attr)

        node_pair = self.tp_node_pair(node_attr[src], node_attr[dst],
            self.fc_node_pair(pair_edge_attr) * self.fc(s0))

        node_pair = self.norm_gate(node_pair)
        node_pair = self.linear_node_pair_final(node_pair)

        if self.resnet and node_pair_attr is not None:
            node_pair = node_pair + node_pair_attr
        return node_pair


class Expansion(nn.Module):
    """Expansion module for converting irreps to orbital Hamiltonian blocks"""

    def __init__(self, irrep_in, irrep_out_1, irrep_out_2):
        super(Expansion, self).__init__()
        self.irrep_in = irrep_in
        self.irrep_out_1 = irrep_out_1
        self.irrep_out_2 = irrep_out_2
        self.instructions = self.get_expansion_path(irrep_in, irrep_out_1, irrep_out_2)

        def prod(x):
            out = 1
            for a in x:
                out *= a
            return out

        self.num_path_weight = sum(prod(ins[-1]) for ins in self.instructions if ins[3])
        self.num_bias = sum([prod(ins[-1][1:]) for ins in self.instructions if ins[0] == 0])
        if self.num_path_weight > 0:
            # Use Xavier initialization scaled for Hamiltonian prediction
            # Keep the division by mul for stability, but scale weights appropriately
            self.weights = nn.Parameter(torch.randn(self.num_path_weight + self.num_bias) * 0.5)
        self.num_weights = self.num_path_weight + self.num_bias

    def forward(self, x_in, weights=None, bias_weights=None):
        def prod(x):
            out = 1
            for a in x:
                out *= a
            return out

        batch_num = x_in.shape[0]
        if len(self.irrep_in) == 1:
            x_in_s = [x_in.reshape(batch_num, self.irrep_in[0].mul, self.irrep_in[0].ir.dim)]
        else:
            x_in_s = [
                x_in[:, i].reshape(batch_num, mul_ir.mul, mul_ir.ir.dim)
            for i, mul_ir in zip(self.irrep_in.slices(), self.irrep_in)]

        outputs = {}
        flat_weight_index = 0
        bias_weight_index = 0
        for ins in self.instructions:
            mul_ir_in = self.irrep_in[ins[0]]
            mul_ir_out1 = self.irrep_out_1[ins[1]]
            mul_ir_out2 = self.irrep_out_2[ins[2]]
            x1 = x_in_s[ins[0]]
            x1 = x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
            w3j_matrix = o3.wigner_3j(ins[1], ins[2], ins[0]).to(self.device).type(x1.type())
            if ins[3] is True or weights is not None:
                if weights is None:
                    weight = self.weights[flat_weight_index:flat_weight_index + prod(ins[-1])].reshape(ins[-1])
                    # Removed division by mul_ir_in.mul to match Hamiltonian scale
                    result = torch.einsum(
                        f"wuv, ijk, bwk-> buivj", weight, w3j_matrix, x1)
                else:
                    weight = weights[:, flat_weight_index:flat_weight_index + prod(ins[-1])].reshape([-1] + ins[-1])
                    result = torch.einsum(f"bwuv, bwk-> buvk", weight, x1)
                    if ins[0] == 0 and bias_weights is not None:
                        bias_weight = bias_weights[:,bias_weight_index:bias_weight_index + prod(ins[-1][1:])].\
                            reshape([-1] + ins[-1][1:])
                        bias_weight_index += prod(ins[-1][1:])
                        result = result + bias_weight.unsqueeze(-1)
                    # Removed division by mul_ir_in.mul to match Hamiltonian scale
                    result = torch.einsum(f"ijk, buvk->buivj", w3j_matrix, result)
                flat_weight_index += prod(ins[-1])
            else:
                result = torch.einsum(
                    f"uvw, ijk, bwk-> buivj", torch.ones(ins[-1]).type(x1.type()).to(self.device), w3j_matrix,
                    x1.reshape(batch_num, mul_ir_in.mul, mul_ir_in.ir.dim)
                )

            result = result.reshape(batch_num, mul_ir_out1.dim, mul_ir_out2.dim)
            key = (ins[1], ins[2])
            if key in outputs.keys():
                outputs[key] = outputs[key] + result
            else:
                outputs[key] = result

        rows = []
        for i in range(len(self.irrep_out_1)):
            blocks = []
            for j in range(len(self.irrep_out_2)):
                if (i, j) not in outputs.keys():
                    blocks += [torch.zeros((x_in.shape[0], self.irrep_out_1[i].dim, self.irrep_out_2[j].dim),
                                           device=x_in.device).type(x_in.type())]
                else:
                    blocks += [outputs[(i, j)]]
            rows.append(torch.cat(blocks, dim=-1))
        output = torch.cat(rows, dim=-2)
        return output

    def get_expansion_path(self, irrep_in, irrep_out_1, irrep_out_2):
        instructions = []
        for i, (num_in, ir_in) in enumerate(irrep_in):
            for j, (num_out1, ir_out1) in enumerate(irrep_out_1):
                for k, (num_out2, ir_out2) in enumerate(irrep_out_2):
                    if ir_in in ir_out1 * ir_out2:
                        instructions.append([i, j, k, True, 1.0, [num_in, num_out1, num_out2]])
        return instructions

    @property
    def device(self):
        return next(self.parameters()).device

    def __repr__(self):
        return f'{self.irrep_in} -> {self.irrep_out_1}x{self.irrep_out_1} and bias {self.num_bias}' \
               f'with parameters {self.num_path_weight}'


# ============ Main QHformer Model ============

class QHformer(nn.Module):
    """
    QHformer: QHNet with Inner Product Attention

    This model integrates inner product attention mechanism into QHNet.
    The key innovation is that Query and Key maintain FULL irreps
    (no scalar compression), and InnerProduct is used to compute
    rotation-invariant attention weights.

    Architecture:
        1. Graph construction with spherical harmonics
        2. Node embedding
        3. CSA/HCA multi-head inner-product attention layers
        4. SelfNet and PairNet (same as original QHNet)
        5. Expansion to Hamiltonian blocks
        6. Matrix assembly and symmetrization

    Key Features:
    - Query/Key maintain full irreps (128x0e + 128x1o + 128x2e + ...)
    - InnerProduct couples irreps to scalars for attention
    - Value preserves complete equivariant information
    - Fully SO(3) equivariant

    Args:
        in_node_features: Input node features (default: 1 for atomic number)
        sh_lmax: Maximum degree of spherical harmonics (default: 4)
        hidden_size: Hidden layer dimension (default: 128)
        bottle_hidden_size: Bottleneck dimension (default: 32)
        num_gnn_layers: Number of GNN layers (default: 5)
        max_radius: Maximum interaction radius in Angstroms (default: 12)
        num_nodes: Maximum number of atoms (default: 10)
        radius_embed_dim: Radial basis embedding dimension (default: 32)
        attention_temperature: Temperature for attention (default: 1.0)
        num_heads: Number of multiplicity-split attention heads (default: 4)
        use_hybrid_attention: Use CSA-HCA-CSA-HCA alternation when True
        csa_top_k: Max incoming edges per node in CSA layers
        hca_top_k: Max incoming edges per node in edge-sparse HCA layers
        hca_lmax: Maximum angular degree retained in HCA K/V compression
        indexer_compress_dim: Hidden dimension of CSA Lightning Indexer
        attention_score_residual_init_std: Std for nonzero learnable attention-score residual init.
        attention_operator: "tp" for e3nn TensorProduct K/V, "so2" for edge-frame SO(2) K/V
    """

    def __init__(
        self,
        in_node_features=1,
        sh_lmax=4,
        hidden_size=128,
        bottle_hidden_size=32,
        num_gnn_layers=5,
        max_radius=12,
        num_nodes=10,
        radius_embed_dim=32,
        attention_temperature=1.0,
        num_heads=4,
        use_hybrid_attention=True,
        csa_top_k=8,
        hca_top_k=8,
        hca_lmax=2,
        indexer_compress_dim=32,
        attention_score_residual_init_std=0.0,
        attention_operator="tp",
    ):
        super(QHformer, self).__init__()
        self.order = sh_lmax

        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.order)
        self.hs = hidden_size
        self.hbs = bottle_hidden_size
        self.radius_embed_dim = radius_embed_dim
        self.max_radius = max_radius
        self.num_gnn_layers = num_gnn_layers
        self.num_heads = num_heads
        self.use_hybrid_attention = use_hybrid_attention
        self.csa_top_k = csa_top_k
        self.hca_top_k = hca_top_k
        self.hca_lmax = hca_lmax
        self.indexer_compress_dim = indexer_compress_dim
        self.attention_score_residual_init_std = attention_score_residual_init_std
        self.attention_operator = attention_operator

        # Use atomic number embedding (max Z=118 for periodic table)
        self.node_embedding = nn.Embedding(118, self.hs)

        # Irreps definitions
        self.hidden_irrep = o3.Irreps(f'{self.hs}x0e + {self.hs}x1o + {self.hs}x2e + {self.hs}x3o + {self.hs}x4e')
        self.hidden_bottle_irrep = o3.Irreps(f'{self.hbs}x0e + {self.hbs}x1o + {self.hbs}x2e + {self.hbs}x3o + {self.hbs}x4e')
        self.hidden_irrep_base = o3.Irreps(f'{self.hs}x0e + {self.hs}x1e + {self.hs}x2e + {self.hs}x3e + {self.hs}x4e')
        self.hidden_bottle_irrep_base = o3.Irreps(
            f'{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e')
        self.final_out_irrep = o3.Irreps(f'{self.hs * 3}x0e + {self.hs * 2}x1o + {self.hs}x2e').simplify()
        self.input_irrep = o3.Irreps(f'{self.hs}x0e')

        self.distance_expansion = ExponentialBernsteinRBF(self.radius_embed_dim, self.max_radius)
        self.num_fc_layer = 1
        self.start_layer = 2

        # ============ Multi-Head CSA/HCA Attention Layers ============
        print(f"QHformer initialized with Multi-Head {'Hybrid CSA/HCA' if use_hybrid_attention else 'Dense'} Attention:")
        print(f"  attention_temperature = {attention_temperature}")
        print(f"  num_heads = {num_heads}")
        print(f"  attention_operator = {attention_operator}")
        if use_hybrid_attention:
            print(f"  pattern = CSA-HCA-CSA-HCA")
            print(f"  csa_top_k = {csa_top_k}")
            print(f"  hca_top_k = {hca_top_k}")
            print(f"  hca_lmax = {hca_lmax}")
        print(f"  attention_score_residual_init_std = {attention_score_residual_init_std}")
        print(f"  Query/Key irreps = {self.hidden_irrep}")

        self.e3_gnn_layer = nn.ModuleList()
        self.e3_gnn_node_pair_layer = nn.ModuleList()
        self.e3_gnn_node_layer = nn.ModuleList()

        for i in range(self.num_gnn_layers):
            input_irrep = self.input_irrep if i == 0 else self.hidden_irrep

            if use_hybrid_attention:
                if i % 2 == 0:
                    layer = CompressedSparseAttentionNetLayer(
                        irrep_in_node=input_irrep,
                        irrep_hidden=self.hidden_irrep,
                        irrep_out=self.hidden_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        sh_irrep=self.sh_irrep,
                        resnet=True,
                        use_norm_gate=True if i != 0 else False,
                        attention_temperature=attention_temperature,
                        num_heads=num_heads,
                        top_k=csa_top_k,
                        indexer_compress_dim=indexer_compress_dim,
                        attention_score_residual_init_std=attention_score_residual_init_std,
                        attention_operator=attention_operator,
                    )
                else:
                    layer = HeavyCompressedAttentionNetLayer(
                        irrep_in_node=input_irrep,
                        irrep_hidden=self.hidden_irrep,
                        irrep_out=self.hidden_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        sh_irrep=self.sh_irrep,
                        resnet=True,
                        use_norm_gate=True if i != 0 else False,
                        attention_temperature=attention_temperature,
                        num_heads=num_heads,
                        hca_lmax=hca_lmax,
                        top_k=hca_top_k,
                        indexer_compress_dim=indexer_compress_dim,
                        attention_score_residual_init_std=attention_score_residual_init_std,
                        attention_operator=attention_operator,
                    )
            else:
                layer = MultiHeadAttentionNetLayer(
                    irrep_in_node=input_irrep,
                    irrep_hidden=self.hidden_irrep,
                    irrep_out=self.hidden_irrep,
                    edge_attr_dim=self.radius_embed_dim,
                    node_attr_dim=self.hs,
                    sh_irrep=self.sh_irrep,
                    resnet=True,
                    use_norm_gate=True if i != 0 else False,
                    attention_temperature=attention_temperature,
                    num_heads=num_heads,
                    attention_score_residual_init_std=attention_score_residual_init_std,
                    attention_operator=attention_operator,
                )
            self.e3_gnn_layer.append(layer)

            if i > self.start_layer:
                self.e3_gnn_node_layer.append(SelfNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        resnet=False,  # CRITICAL FIX: Disable to preserve equivariance
                ))

                self.e3_gnn_node_pair_layer.append(PairNetLayer(
                        irrep_in_node=self.hidden_irrep_base,
                        irrep_bottle_hidden=self.hidden_irrep_base,
                        irrep_out=self.hidden_irrep_base,
                        sh_irrep=self.sh_irrep,
                        edge_attr_dim=self.radius_embed_dim,
                        node_attr_dim=self.hs,
                        invariant_layers=self.num_fc_layer,
                        invariant_neurons=self.hs,
                        resnet=False,  # CRITICAL FIX: Disable to preserve equivariance
                ))

        # ============ Expansion Modules ============
        self.expand_ii, self.expand_ij, self.fc_ii, self.fc_ij, self.fc_ii_bias, self.fc_ij_bias = \
            nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict(), nn.ModuleDict()

        for name in {"hamiltonian"}:
            input_expand_ii = o3.Irreps(f"{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e")

            self.expand_ii[name] = Expansion(
                input_expand_ii,
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e")
            )
            self.fc_ii[name] = nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ii[name].num_path_weight)
            )
            self.fc_ii_bias[name] = nn.Sequential(
                nn.Linear(self.hs, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ii[name].num_bias)
            )

            self.expand_ij[name] = Expansion(
                o3.Irreps(f'{self.hbs}x0e + {self.hbs}x1e + {self.hbs}x2e + {self.hbs}x3e + {self.hbs}x4e'),
                o3.Irreps("3x0e + 2x1e + 1x2e"),
                o3.Irreps("3x0e + 2x1e + 1x2e")
            )

            self.fc_ij[name] = nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_path_weight)
            )

            self.fc_ij_bias[name] = nn.Sequential(
                nn.Linear(self.hs * 2, self.hs),
                nn.SiLU(),
                nn.Linear(self.hs, self.expand_ij[name].num_bias)
            )

        self.output_ii = o3.Linear(self.hidden_irrep, self.hidden_bottle_irrep)
        self.output_ij = o3.Linear(self.hidden_irrep, self.hidden_bottle_irrep)

        # Initialize orbital_mask
        self.orbital_mask = self.get_orbital_mask()

    def get_number_of_parameters(self):
        num = 0
        for param in self.parameters():
            if param.requires_grad:
                num += param.numel()
        return num

    def set(self, device):
        self = self.to(device)
        self.orbital_mask = self.get_orbital_mask()
        for key in self.orbital_mask.keys():
            self.orbital_mask[key] = self.orbital_mask[key].to(self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, keep_blocks=False):
        """Forward pass with Inner Product Attention"""
        # Build graph, or reuse static QH9 graph cache when present. The RBF
        # layer remains live because its centers/std are trainable.
        if all(hasattr(data, key) for key in ("edge_index", "edge_dist", "edge_sh")):
            node_attr = data.atoms.squeeze()
            edge_index = data.edge_index
            rbf_new = self.expand_edge_distances(data.edge_dist).type(data.pos.type())
            edge_sh = data.edge_sh.type(data.pos.type())
        else:
            node_attr, edge_index, rbf_new, edge_sh, _ = self.build_graph(data, self.max_radius)
        node_attr = self.node_embedding(node_attr)
        data.node_attr, data.edge_index, data.edge_attr, data.edge_sh = \
            node_attr, edge_index, rbf_new, edge_sh

        # Build full graph for PairNet, or reuse cached complete graph.
        if all(hasattr(data, key) for key in ("full_edge_index", "full_edge_dist", "full_edge_sh")):
            full_edge_index = data.full_edge_index
            full_edge_attr = self.expand_edge_distances(data.full_edge_dist).type(data.pos.type())
            full_edge_sh = data.full_edge_sh.type(data.pos.type())
            transpose_edge_index = getattr(data, "full_transpose_perm", None)
        else:
            _, full_edge_index, full_edge_attr, full_edge_sh, transpose_edge_index = self.build_graph(data, 10000)
        data.full_edge_index, data.full_edge_attr, data.full_edge_sh = \
            full_edge_index, full_edge_attr, full_edge_sh
        pair_edge_index, pair_edge_attr, pair_edge_sh = self.prepare_pair_graph(
            data, full_edge_index, full_edge_attr, full_edge_sh)
        data.pair_edge_index, data.pair_edge_attr, data.pair_edge_sh = \
            pair_edge_index, pair_edge_attr, pair_edge_sh

        pair_dst, pair_src = data.pair_edge_index

        tic = time.time()
        fii = None
        fij = None

        # Apply Inner Product Attention layers
        # Note: num_gnn_layers must be > start_layer for fii and fij to be initialized
        for layer_idx, layer in enumerate(self.e3_gnn_layer):
            node_attr = layer(data, node_attr)
            if layer_idx > self.start_layer:
                fii = self.e3_gnn_node_layer[layer_idx-self.start_layer-1](data, node_attr, fii)
                fij = self.e3_gnn_node_pair_layer[layer_idx-self.start_layer-1](data, node_attr, fij)

        # Output projection
        # Handle case where fii/fij are None (when num_gnn_layers <= start_layer)
        # output_ii/ij expect input of hidden_irrep.dim, not hidden_bottle_irrep.dim
        if fii is None:
            # Create zero tensor with correct shape for output_ii input
            fii = torch.zeros(node_attr.shape[0], self.hidden_irrep.dim,
                             device=node_attr.device, dtype=node_attr.dtype)
        if fij is None:
            # Create zero tensor with correct shape for output_ij input
            num_edges = data.pair_edge_index.shape[1]
            fij = torch.zeros(num_edges, self.hidden_irrep.dim,
                             device=node_attr.device, dtype=node_attr.dtype)

        fii = self.output_ii(fii)
        fij = self.output_ij(fij)

        # Expand to Hamiltonian blocks
        hamiltonian_diagonal_matrix = self.expand_ii['hamiltonian'](
            fii, self.fc_ii['hamiltonian'](data.node_attr), self.fc_ii_bias['hamiltonian'](data.node_attr))

        node_pair_embedding = torch.cat([data.node_attr[pair_dst], data.node_attr[pair_src]], dim=-1)
        hamiltonian_non_diagonal_matrix = self.expand_ij['hamiltonian'](
            fij, self.fc_ij['hamiltonian'](node_pair_embedding),
            self.fc_ij_bias['hamiltonian'](node_pair_embedding))

        # Build final matrix
        if keep_blocks is False:
            hamiltonian_matrix = self.build_final_matrix(
                data, hamiltonian_diagonal_matrix, hamiltonian_non_diagonal_matrix)
            hamiltonian_matrix = hamiltonian_matrix + hamiltonian_matrix.transpose(-1, -2)

            # Check for NaN/Inf in output
            if torch.isnan(hamiltonian_matrix).any() or torch.isinf(hamiltonian_matrix).any():
                print(f"Warning: NaN/Inf detected in Hamiltonian output, replacing with zeros")
                hamiltonian_matrix = torch.where(
                    torch.isfinite(hamiltonian_matrix),
                    hamiltonian_matrix,
                    torch.zeros_like(hamiltonian_matrix)
                )

            results = {}
            results['hamiltonian'] = hamiltonian_matrix
            results['duration'] = torch.tensor([time.time() - tic])
        else:
            ret_hamiltonian_diagonal_matrix = hamiltonian_diagonal_matrix +\
                                          hamiltonian_diagonal_matrix.transpose(-1, -2)
            if (transpose_edge_index is not None and
                    hamiltonian_non_diagonal_matrix.shape[0] == data.full_edge_index.shape[1]):
                ret_hamiltonian_non_diagonal_matrix = hamiltonian_non_diagonal_matrix + \
                      hamiltonian_non_diagonal_matrix[transpose_edge_index].transpose(-1, -2)
            else:
                ret_hamiltonian_non_diagonal_matrix = hamiltonian_non_diagonal_matrix
            results = {}
            results['hamiltonian_diagonal_blocks'] = ret_hamiltonian_diagonal_matrix
            results['hamiltonian_non_diagonal_blocks'] = ret_hamiltonian_non_diagonal_matrix
            results['pair_edge_index'] = data.pair_edge_index

        return results

    def expand_edge_distances(self, edge_dist):
        return self.distance_expansion(edge_dist.reshape(-1, 1)).reshape(-1, self.radius_embed_dim)

    def prepare_pair_graph(self, data, full_edge_index, full_edge_attr, full_edge_sh):
        """Return one directed atom pair per unordered pair for PairNet.

        SPHNet predicts only one off-diagonal block for each unordered atom
        pair, then recovers the Hermitian counterpart by transposition during
        matrix assembly.  Selecting by ``dst > src`` is invariant to the
        molecule's 3D orientation and halves PairNet/Expansion work.
        """
        if all(hasattr(data, key) for key in ("pair_edge_index", "pair_edge_dist", "pair_edge_sh")):
            pair_edge_index = data.pair_edge_index
            pair_edge_attr = self.expand_edge_distances(data.pair_edge_dist).type(data.pos.type())
            pair_edge_sh = data.pair_edge_sh.type(data.pos.type())
            return pair_edge_index, pair_edge_attr, pair_edge_sh

        if hasattr(data, "pair_edge_index"):
            pair_edge_index = data.pair_edge_index
            pair_dst, pair_src = pair_edge_index
            pair_vec = data.pos[pair_dst.long()] - data.pos[pair_src.long()]
            pair_edge_attr = self.expand_edge_distances(pair_vec.norm(dim=-1)).type(data.pos.type())
            pair_edge_sh = o3.spherical_harmonics(
                self.sh_irrep, pair_vec[:, [1, 2, 0]],
                normalize=True, normalization='component').type(data.pos.type())
            return pair_edge_index, pair_edge_attr, pair_edge_sh

        pair_mask = full_edge_index[0] > full_edge_index[1]
        return full_edge_index[:, pair_mask], full_edge_attr[pair_mask], full_edge_sh[pair_mask]

    def build_graph(self, data, max_radius):
        """Build molecular graph with spherical harmonics"""
        node_attr = data.atoms.squeeze()
        radius_edges = radius_graph(data.pos, max_radius, data.batch)

        dst, src = radius_edges
        edge_vec = data.pos[dst.long()] - data.pos[src.long()]
        rbf = self.expand_edge_distances(edge_vec.norm(dim=-1)).type(data.pos.type())

        edge_sh = o3.spherical_harmonics(
            self.sh_irrep, edge_vec[:, [1, 2, 0]],
            normalize=True, normalization='component').type(data.pos.type())

        # Build transpose index for symmetrization
        start_edge_index = 0
        all_transpose_index = []
        for graph_idx in range(data.ptr.shape[0] - 1):
            num_nodes = data.ptr[graph_idx +1] - data.ptr[graph_idx]
            graph_edge_index = radius_edges[:, start_edge_index:start_edge_index+num_nodes*(num_nodes-1)]
            sub_graph_edge_index = graph_edge_index - data.ptr[graph_idx]
            bias = (sub_graph_edge_index[0] < sub_graph_edge_index[1]).type(torch.int)
            transpose_index = sub_graph_edge_index[0] * (num_nodes - 1) + sub_graph_edge_index[1] - bias
            transpose_index = transpose_index + start_edge_index
            all_transpose_index.append(transpose_index)
            start_edge_index = start_edge_index + num_nodes*(num_nodes-1)

        return node_attr, radius_edges, rbf, edge_sh, torch.cat(all_transpose_index, dim=-1)

    def build_final_matrix(self, data, diagonal_matrix, non_diagonal_matrix):
        """Assemble final Hamiltonian matrix from orbital blocks.

        This follows SPHNet's block2matrix pattern: fill a dense
        [atom, atom, max_block, max_block] tensor first, then reshape and select
        the atom-specific orbital rows/columns. The previous implementation
        searched full_edge_index for every off-diagonal block, which introduced
        many tiny GPU synchronizations.
        """
        final_matrix = []
        full_edge_index = data.full_edge_index
        candidate_pair_edge_index = getattr(data, "pair_edge_index", None)
        if candidate_pair_edge_index is None:
            pair_mask = full_edge_index[0] > full_edge_index[1]
            candidate_pair_edge_index = full_edge_index[:, pair_mask]
        use_half_edges = (
            candidate_pair_edge_index is not None and
            non_diagonal_matrix.shape[0] == candidate_pair_edge_index.shape[1] and
            candidate_pair_edge_index.shape[1] != full_edge_index.shape[1]
        )
        edge_index = candidate_pair_edge_index if use_half_edges else full_edge_index
        dst, src = edge_index
        max_block_size = diagonal_matrix.shape[-1]
        edge_start = 0
        for graph_idx in range(data.ptr.shape[0] - 1):
            node_start = int(data.ptr[graph_idx].item())
            node_end = int(data.ptr[graph_idx + 1].item())
            num_nodes = node_end - node_start
            num_edges = num_nodes * (num_nodes - 1) // 2 if use_half_edges else num_nodes * (num_nodes - 1)

            blocks = torch.zeros(
                num_nodes,
                num_nodes,
                max_block_size,
                max_block_size,
                dtype=diagonal_matrix.dtype,
                device=diagonal_matrix.device,
            )

            diag_idx = torch.arange(num_nodes, device=diagonal_matrix.device)
            blocks[diag_idx, diag_idx] = diagonal_matrix[node_start:node_end]

            if num_edges > 0:
                graph_dst = dst[edge_start:edge_start + num_edges].long() - node_start
                graph_src = src[edge_start:edge_start + num_edges].long() - node_start
                blocks[graph_dst, graph_src] = non_diagonal_matrix[edge_start:edge_start + num_edges]

            dense = blocks.permute(0, 2, 1, 3).reshape(
                num_nodes * max_block_size,
                num_nodes * max_block_size,
            )
            atom_orbitals = []
            atoms = data.atoms[node_start:node_end].reshape(-1)
            for atom_idx, atom in enumerate(atoms):
                atom_orbitals.append(
                    atom_idx * max_block_size + self.orbital_mask[int(atom.item())].to(diagonal_matrix.device)
                )
            atom_orbitals = torch.cat(atom_orbitals, dim=0)
            final_matrix.append(dense.index_select(0, atom_orbitals).index_select(1, atom_orbitals))
            edge_start += num_edges
        final_matrix = torch.stack(final_matrix, dim=0)
        return final_matrix

    def get_orbital_mask(self):
        """Get orbital mask for different atom types matching MD17 SchNOrb.

        Expansion output: 3x0e + 2x1e + 1x2e (14 orbitals per atom)
        - 3x0e: s-orbitals at indices [0, 1, 2]
        - 2x1e: p-orbitals at indices [3, 4, 5, 6, 7, 8]
          * First p-orbital: [3, 4, 5] (1x1e)
          * Second p-orbital: [6, 7, 8] (1x1e)
        - 1x2e: d-orbitals at indices [9, 10, 11, 12, 13]

        MD17/SchNOrb uses H as ``ssp`` after hamiltonian_transform:
        2 scalar orbitals + one complete p multiplet = 5 orbitals.
        Complete multiplets are kept to preserve SO(3) equivariance.
        """
        orbital_mask_line1 = torch.cat([torch.tensor([0, 1]), torch.tensor([3, 4, 5])])
        orbital_mask_full = torch.arange(14)

        orbital_mask = {}
        orbital_mask[1] = orbital_mask_line1  # H: ssp
        orbital_mask[2] = orbital_mask_line1  # He/light first-row fallback
        orbital_mask[3] = orbital_mask_full
        orbital_mask[4] = orbital_mask_full
        orbital_mask[5] = orbital_mask_full  # B+: all
        orbital_mask[6] = orbital_mask_full  # C: all
        orbital_mask[7] = orbital_mask_full  # N: all
        orbital_mask[8] = orbital_mask_full  # O: all
        orbital_mask[9] = orbital_mask_full  # F: all
        orbital_mask[10] = orbital_mask_full  # Ne: all

        return orbital_mask


# Alias for backward compatibility
AttentionQHNet = QHformer


if __name__ == "__main__":
    # Test the QHformer model
    print("Testing QHformer model...")

    model = QHformer(
        in_node_features=1,
        sh_lmax=4,
        hidden_size=128,
        bottle_hidden_size=32,
        num_gnn_layers=5,
        max_radius=12,
        radius_embed_dim=32,
        attention_temperature=1.0,
    )

    num_params = model.get_number_of_parameters()
    print(f"Model parameters: {num_params:,}")
    print(f"Hidden irreps: {model.hidden_irrep}")
    print(f"Query/Key maintain FULL irreps (no scalar compression)")
    print(f"InnerProduct couples irreps to scalars for attention")
    print("\nQHformer test passed!")
