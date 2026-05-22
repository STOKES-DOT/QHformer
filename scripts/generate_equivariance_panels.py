"""
Generate standalone Hamiltonian equivariance diagnostic panels for README.

The panels use a randomly initialized, small QHformer v2 model on a single
water molecule. They are architecture diagnostics, not training results.
"""

import argparse
import os
import sys
import types

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(REPO_ROOT, ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from e3nn import o3
from torch_geometric.data import Data


sys.path.insert(0, REPO_ROOT)


def _install_radius_graph_fallback():
    try:
        from torch_cluster import radius_graph  # noqa: F401
        return
    except Exception:
        pass

    def radius_graph_pure(pos, r, batch=None, loop=False, max_num_neighbors=32, flow="source_to_target"):
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        edge_indices = []
        batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
        for graph_idx in range(batch_size):
            node_mask = batch == graph_idx
            node_idx = torch.nonzero(node_mask, as_tuple=False).squeeze(-1)
            node_pos = pos[node_idx]
            if node_pos.numel() == 0:
                continue
            dist = torch.cdist(node_pos, node_pos)
            adj = dist <= r
            if not loop:
                adj.fill_diagonal_(False)
            edge_index = adj.nonzero(as_tuple=False).t().contiguous()
            if edge_index.numel() > 0:
                edge_index[0] = node_idx[edge_index[0]]
                edge_index[1] = node_idx[edge_index[1]]
                edge_indices.append(edge_index)
        if edge_indices:
            return torch.cat(edge_indices, dim=1)
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)

    torch_cluster = types.ModuleType("torch_cluster")
    torch_cluster.radius_graph = radius_graph_pure
    sys.modules["torch_cluster"] = torch_cluster


_install_radius_graph_fallback()

from models.qhformer import QHformer  # noqa: E402


def make_water_data(pos):
    return Data(
        pos=pos,
        atoms=torch.tensor([[8], [1], [1]], dtype=torch.long),
        batch=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3], dtype=torch.long),
    )


def make_model(seed):
    torch.manual_seed(seed)
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
        hca_lmax=3,
        indexer_compress_dim=8,
        attention_score_residual_init_std=0.0,
    )
    model.eval()
    model.set("cpu")
    return model


def save_matrix(path, matrix, title, cmap, vmin, vmax, cbar_label):
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=180)
    image = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=13, pad=8)
    ax.set_xlabel("AO index")
    ax.set_ylabel("AO index")
    ax.set_xticks([0, 5, 10, 15, 20, 23])
    ax.set_yticks([0, 5, 10, 15, 20, 23])
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_spectrum(path, eig_original, eig_rotated):
    eig_delta = (eig_rotated - eig_original).abs()
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 4.0), dpi=180, gridspec_kw={"height_ratios": [3, 1]})

    x = torch.arange(eig_original.numel())
    axes[0].plot(x, eig_original, marker="o", linewidth=1.6, markersize=3, label="H(X)")
    axes[0].plot(x, eig_rotated, marker="x", linewidth=1.2, markersize=3, label="H(RX)")
    axes[0].set_title("Spectrum Invariance", fontsize=13, pad=8)
    axes[0].set_ylabel("Eigenvalue")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    axes[1].semilogy(x, eig_delta.clamp_min(1e-12), color="black", linewidth=1.4)
    axes[1].set_xlabel("Eigenvalue index")
    axes[1].set_ylabel("|Δλ|")
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "images", "equivariance"))
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    pos = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [0.9572, 0.0000, 0.0000],
            [-0.2390, 0.9270, 0.0000],
        ],
        dtype=torch.float32,
    )
    rotation = o3.rand_matrix().to(pos.dtype)
    model = make_model(args.seed)

    with torch.no_grad():
        h_original = model(make_water_data(pos))["hamiltonian"][0]
        h_rotated = model(make_water_data(pos @ rotation.T))["hamiltonian"][0]

    delta = h_rotated - h_original
    abs_delta = delta.abs()
    eig_original = torch.linalg.eigvalsh(h_original)
    eig_rotated = torch.linalg.eigvalsh(h_rotated)

    h_limit = torch.stack([h_original.abs().max(), h_rotated.abs().max()]).max().item()
    d_limit = abs_delta.max().item()

    save_matrix(
        os.path.join(args.output_dir, "h_original.png"),
        h_original.numpy(),
        "H(X)",
        "RdBu_r",
        -h_limit,
        h_limit,
        "Hartree",
    )
    save_matrix(
        os.path.join(args.output_dir, "h_rotated.png"),
        h_rotated.numpy(),
        "H(RX)",
        "RdBu_r",
        -h_limit,
        h_limit,
        "Hartree",
    )
    save_matrix(
        os.path.join(args.output_dir, "h_delta_abs.png"),
        abs_delta.numpy(),
        "|H(RX) - H(X)|",
        "inferno",
        0.0,
        d_limit,
        "Abs. Hartree",
    )
    save_spectrum(
        os.path.join(args.output_dir, "spectrum_invariance.png"),
        eig_original,
        eig_rotated,
    )

    metrics = {
        "max_abs_matrix_change": abs_delta.max().item(),
        "mae_matrix_change": abs_delta.mean().item(),
        "trace_abs_diff": (torch.trace(h_rotated) - torch.trace(h_original)).abs().item(),
        "max_abs_eigenvalue_diff": (eig_rotated - eig_original).abs().max().item(),
    }
    with open(os.path.join(args.output_dir, "metrics.txt"), "w", encoding="utf-8") as handle:
        for key, value in metrics.items():
            handle.write(f"{key}: {value:.8e}\n")

    for key, value in metrics.items():
        print(f"{key}: {value:.8e}")


if __name__ == "__main__":
    main()
