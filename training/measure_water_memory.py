"""
Measure QHformer training memory on MD17/SchNOrb water samples.

This script reports peak CUDA memory for forward + backward + optimizer step
at several batch sizes, then fits a linear estimate of marginal memory per
water molecule.
"""

import argparse
import gc
import os
import sys
import types

import torch
from torch_geometric.data import Batch, Data


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/yjiao/QHformer/dataset")
    parser.add_argument("--molecule", default="water")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128,256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--bottle-hidden-size", type=int, default=64)
    parser.add_argument("--num-gnn-layers", type=int, default=4)
    parser.add_argument("--radius-embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--csa-top-k", type=int, default=8)
    parser.add_argument("--hca-lmax", type=int, default=2)
    parser.add_argument("--indexer-compress-dim", type=int, default=32)
    parser.add_argument("--attention-score-residual-init-std", type=float, default=0.0)
    parser.add_argument(
        "--optimizer-lr",
        type=float,
        default=0.0,
        help="Use 0.0 to allocate optimizer state without changing random parameters.",
    )
    return parser.parse_args()


def load_samples(data_root, molecule, num_samples):
    path = os.path.join(data_root, molecule, "processed", "data.pt")
    data, slices = torch.load(path, weights_only=False)
    available = int(slices["pos"].numel() - 1)
    if num_samples > available:
        raise ValueError(f"Requested {num_samples} samples, but only {available} are available")

    samples = []
    for idx in range(num_samples):
        samples.append(
            Data(
                pos=data.pos[slices["pos"][idx]:slices["pos"][idx + 1]].float(),
                atoms=data.atoms[slices["atoms"][idx]:slices["atoms"][idx + 1]].long(),
                hamiltonian=data.hamiltonian[
                    slices["hamiltonian"][idx]:slices["hamiltonian"][idx + 1]
                ].float(),
            )
        )
    return samples


def make_model(args, device):
    model = QHformer(
        in_node_features=4,
        sh_lmax=4,
        hidden_size=args.hidden_size,
        bottle_hidden_size=args.bottle_hidden_size,
        num_gnn_layers=args.num_gnn_layers,
        max_radius=12,
        num_nodes=10,
        radius_embed_dim=args.radius_embed_dim,
        attention_temperature=1.0,
        num_heads=args.num_heads,
        use_hybrid_attention=True,
        csa_top_k=args.csa_top_k,
        hca_lmax=args.hca_lmax,
        indexer_compress_dim=args.indexer_compress_dim,
        attention_score_residual_init_std=args.attention_score_residual_init_std,
    ).to(device)
    model.set(device)
    return model


def train_step(model, optimizer, samples, batch_size, device):
    batch = Batch.from_data_list(samples[:batch_size]).to(device)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)["hamiltonian"]
    target = batch.hamiltonian.view(output.shape[0], output.shape[1], output.shape[2])
    pred = 0.5 * (output + output.transpose(-1, -2))
    target = 0.5 * (target + target.transpose(-1, -2))
    diff = pred - target
    mae = diff.abs().mean()
    mse = (diff ** 2).mean()
    loss = mae + mse
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    return loss.detach().item(), mae.detach().item(), mse.detach().item()


def mib(value):
    return value / (1024 ** 2)


def main():
    args = parse_args()
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    max_batch_size = max(batch_sizes)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for memory measurement")
    torch.manual_seed(args.seed)
    torch.cuda.set_device(torch.device(args.device))
    device = torch.device(args.device)

    samples = load_samples(args.data_root, args.molecule, max_batch_size)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model = make_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.optimizer_lr, weight_decay=1e-4)
    torch.cuda.synchronize(device)
    model_alloc = torch.cuda.memory_allocated(device)
    model_reserved = torch.cuda.memory_reserved(device)

    # Allocate Adam state once so later batch-size peaks include realistic optimizer memory.
    train_step(model, optimizer, samples, 1, device)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    optimizer_alloc = torch.cuda.memory_allocated(device)
    optimizer_reserved = torch.cuda.memory_reserved(device)

    print(f"device={device}")
    print(f"num_samples_loaded={len(samples)}")
    print(f"model_allocated_mib={mib(model_alloc):.2f}")
    print(f"model_reserved_mib={mib(model_reserved):.2f}")
    print(f"model_plus_optimizer_allocated_mib={mib(optimizer_alloc):.2f}")
    print(f"model_plus_optimizer_reserved_mib={mib(optimizer_reserved):.2f}")
    print("batch_size,peak_allocated_mib,peak_reserved_mib,loss,mae,mse")

    rows = []
    for batch_size in batch_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            loss, mae, mse = train_step(model, optimizer, samples, batch_size, device)
            torch.cuda.synchronize(device)
            peak_alloc = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            rows.append((batch_size, mib(peak_alloc), mib(peak_reserved), loss, mae, mse))
            print(
                f"{batch_size},{mib(peak_alloc):.2f},{mib(peak_reserved):.2f},"
                f"{loss:.8f},{mae:.8f},{mse:.8f}"
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print(f"{batch_size},OOM,OOM,nan,nan,nan")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            break

    if len(rows) >= 2:
        x = torch.tensor([row[0] for row in rows], dtype=torch.float64)
        y_alloc = torch.tensor([row[1] for row in rows], dtype=torch.float64)
        y_reserved = torch.tensor([row[2] for row in rows], dtype=torch.float64)
        design = torch.stack([torch.ones_like(x), x], dim=1)
        alloc_fit = torch.linalg.lstsq(design, y_alloc).solution
        reserved_fit = torch.linalg.lstsq(design, y_reserved).solution
        print(f"fit_peak_allocated_mib=intercept {alloc_fit[0].item():.2f}, per_molecule {alloc_fit[1].item():.4f}")
        print(f"fit_peak_reserved_mib=intercept {reserved_fit[0].item():.2f}, per_molecule {reserved_fit[1].item():.4f}")


if __name__ == "__main__":
    main()
