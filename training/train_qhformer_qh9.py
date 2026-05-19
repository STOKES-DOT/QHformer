"""
Train QHformer v2 on QH9Stable SQLite samples.

QH9 molecules have variable Hamiltonian sizes, so this script groups samples by
total orbital dimension before batching. Each batch can then be assembled by
QHformer without padding.
"""

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from e3nn import o3
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from models.qhformer import QHformer  # noqa: E402
from models.torch_ops import radius_graph, scatter  # noqa: F401,E402


TORCH_OPS_BACKEND = "torch"


ORBITAL_DIM = {1: 5, 6: 14, 7: 14, 8: 14, 9: 14}


class QH9Data(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_transpose_perm":
            return int(self.edge_index.size(1))
        if key == "full_transpose_perm":
            return int(self.full_edge_index.size(1))
        return super().__inc__(key, value, *args, **kwargs)


class QH9SQLiteHamiltonianDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        db_path,
        max_samples=5000,
        cache_graphs=True,
        graph_cache_path=None,
        max_radius=7.0,
        sh_lmax=4,
    ):
        self.db_path = os.path.abspath(db_path)
        self.max_samples = int(max_samples)
        self.cache_graphs = bool(cache_graphs)
        self.graph_cache_path = Path(graph_cache_path) if graph_cache_path else None
        self.max_radius = float(max_radius)
        self.sh_lmax = int(sh_lmax)
        self.sh_irrep = o3.Irreps.spherical_harmonics(lmax=self.sh_lmax)
        self._conn = None

        conn = sqlite3.connect(self.db_path)
        try:
            total = int(conn.execute("SELECT COUNT(*) FROM data").fetchone()[0])
            rows = conn.execute(
                "SELECT id, N, Z FROM data ORDER BY id LIMIT ?",
                (self.max_samples if self.max_samples > 0 else total,),
            ).fetchall()
        finally:
            conn.close()

        self.total_rows = total
        self.row_ids = [int(row[0]) for row in rows]
        self.n_atoms = [int(row[1]) for row in rows]
        self.atomic_numbers = [
            np.frombuffer(row[2], dtype=np.int32, count=int(row[1])).copy()
            for row in rows
        ]
        self.orbital_dims = [self._expected_orbital_dim(z) for z in self.atomic_numbers]
        self.graph_cache = self._load_or_build_graph_cache() if self.cache_graphs else None

    def _connection(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=OFF")
            self._conn.execute("PRAGMA synchronous=OFF")
            self._conn.execute("PRAGMA temp_store=MEMORY")
        return self._conn

    def _expected_orbital_dim(self, atomic_numbers):
        total = 0
        for atomic_number in atomic_numbers:
            atomic_number = int(atomic_number)
            if atomic_number not in ORBITAL_DIM:
                raise ValueError(f"Unsupported QH9 atomic number: {atomic_number}")
            total += ORBITAL_DIM[atomic_number]
        return total

    def __len__(self):
        return len(self.row_ids)

    def _cache_metadata(self):
        return {
            "db_path": self.db_path,
            "row_ids": self.row_ids,
            "max_radius": self.max_radius,
            "sh_lmax": self.sh_lmax,
        }

    def _load_or_build_graph_cache(self):
        metadata = self._cache_metadata()
        if self.graph_cache_path is not None and self.graph_cache_path.exists():
            cached = torch.load(self.graph_cache_path, map_location="cpu", weights_only=False)
            if cached.get("metadata") == metadata:
                return cached["graphs"]

        graphs = self._build_graph_cache()
        if self.graph_cache_path is not None:
            self.graph_cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"metadata": metadata, "graphs": graphs}, self.graph_cache_path)
        return graphs

    def _build_graph_cache(self):
        conn = sqlite3.connect(self.db_path)
        try:
            pos_rows = []
            for start in range(0, len(self.row_ids), 900):
                chunk = self.row_ids[start:start + 900]
                pos_rows.extend(conn.execute(
                    f"SELECT id, N, pos FROM data WHERE id IN ({','.join('?' for _ in chunk)})",
                    chunk,
                ).fetchall())
        finally:
            conn.close()

        pos_by_id = {
            int(row_id): np.frombuffer(pos_blob, dtype=np.float64, count=int(n_atoms) * 3)
            .copy()
            .reshape(int(n_atoms), 3)
            for row_id, n_atoms, pos_blob in pos_rows
        }
        return [self._build_single_graph_cache(pos_by_id[row_id]) for row_id in self.row_ids]

    def _build_single_graph_cache(self, positions):
        pos = torch.tensor(positions, dtype=torch.float32)
        edge_index, edge_dist, edge_sh = self._make_edges(pos, self.max_radius)
        full_edge_index, full_edge_dist, full_edge_sh = self._make_edges(pos, float("inf"))
        return {
            "edge_index": edge_index,
            "edge_dist": edge_dist,
            "edge_sh": edge_sh,
            "edge_transpose_perm": self._transpose_perm(edge_index),
            "full_edge_index": full_edge_index,
            "full_edge_dist": full_edge_dist,
            "full_edge_sh": full_edge_sh,
            "full_transpose_perm": self._transpose_perm(full_edge_index),
        }

    def _make_edges(self, pos, radius):
        dist = torch.cdist(pos, pos)
        mask = torch.ones_like(dist, dtype=torch.bool) if math.isinf(radius) else dist <= radius
        mask.fill_diagonal_(False)
        edge_index = mask.nonzero(as_tuple=False).t().contiguous()
        if edge_index.numel() == 0:
            return (
                edge_index,
                torch.empty(0, dtype=torch.float32),
                torch.empty(0, self.sh_irrep.dim, dtype=torch.float32),
            )
        dst, src = edge_index
        edge_vec = pos[dst] - pos[src]
        edge_dist = edge_vec.norm(dim=-1).to(torch.float32)
        edge_sh = o3.spherical_harmonics(
            self.sh_irrep,
            edge_vec[:, [1, 2, 0]],
            normalize=True,
            normalization="component",
        ).to(torch.float32)
        return edge_index.to(torch.long), edge_dist, edge_sh

    @staticmethod
    def _transpose_perm(edge_index):
        if edge_index.numel() == 0:
            return torch.empty(0, dtype=torch.long)
        pairs = {
            (int(dst), int(src)): idx
            for idx, (dst, src) in enumerate(edge_index.t().tolist())
        }
        perm = [pairs[(int(src), int(dst))] for dst, src in edge_index.t().tolist()]
        return torch.tensor(perm, dtype=torch.long)

    def __getitem__(self, index):
        row_id = self.row_ids[int(index)]
        row = self._connection().execute(
            "SELECT id, N, Z, pos, Ham FROM data WHERE id=?",
            (row_id,),
        ).fetchone()
        if row is None:
            raise IndexError(f"Missing QH9 row id: {row_id}")

        sample_id, n_atoms, z_blob, pos_blob, ham_blob = row
        n_atoms = int(n_atoms)
        atomic_numbers = np.frombuffer(z_blob, dtype=np.int32, count=n_atoms).copy()
        positions = np.frombuffer(pos_blob, dtype=np.float64, count=n_atoms * 3).copy().reshape(n_atoms, 3)
        num_orbitals = self._expected_orbital_dim(atomic_numbers)
        hamiltonian = np.frombuffer(
            ham_blob,
            dtype=np.float64,
            count=num_orbitals * num_orbitals,
        ).copy().reshape(num_orbitals, num_orbitals)

        data = QH9Data(
            pos=torch.tensor(positions, dtype=torch.float32),
            atoms=torch.tensor(atomic_numbers[:, None], dtype=torch.long),
            hamiltonian=torch.tensor(hamiltonian, dtype=torch.float32),
            sample_id=torch.tensor([int(sample_id)], dtype=torch.long),
            num_nodes=n_atoms,
        )
        if self.graph_cache is not None:
            for key, value in self.graph_cache[int(index)].items():
                setattr(data, key, value)
        return data


def make_bucket_batches(indices, orbital_dims, batch_size, shuffle, seed):
    rng = np.random.default_rng(seed)
    buckets = defaultdict(list)
    for index in indices:
        buckets[int(orbital_dims[int(index)])].append(int(index))

    batches = []
    for _, bucket_indices in sorted(buckets.items()):
        if shuffle:
            rng.shuffle(bucket_indices)
        for start in range(0, len(bucket_indices), batch_size):
            batches.append(bucket_indices[start:start + batch_size])

    if shuffle:
        rng.shuffle(batches)
    return batches


def split_indices(num_samples, train_split, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(indices)
    num_train = int(num_samples * train_split)
    return indices[:num_train].tolist(), indices[num_train:].tolist()


def make_loader(dataset, indices, batch_size, shuffle, seed, num_workers=0):
    batches = make_bucket_batches(indices, dataset.orbital_dims, batch_size, shuffle, seed)
    return DataLoader(dataset, batch_sampler=batches, num_workers=num_workers)


def criterion(outputs, batch):
    h_pred = outputs["hamiltonian"]
    h_true = batch.hamiltonian

    if h_true.dim() == 2 and h_pred.dim() == 3:
        n_orbitals = h_pred.shape[1]
        h_true = h_true.view(batch.num_graphs, n_orbitals, n_orbitals)

    h_pred = 0.5 * (h_pred + h_pred.transpose(-1, -2))
    h_true = 0.5 * (h_true + h_true.transpose(-1, -2))
    diff = h_pred - h_true
    mae = diff.abs().mean()
    mse = diff.pow(2).mean()
    loss = mae + mse
    return loss, mae, torch.sqrt(mse)


def get_lr(epoch, args):
    if epoch <= args.warmup_epochs:
        return args.warmup_start_lr + (args.learning_rate - args.warmup_start_lr) * epoch / args.warmup_epochs
    progress = (epoch - args.warmup_epochs) / max(1, args.num_epochs - args.warmup_epochs)
    return args.min_lr + (args.learning_rate - args.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))


def run_epoch(model, loader, optimizer, device, grad_clip, desc):
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "samples": 0}
    iterator = tqdm(loader, desc=desc, leave=False)

    for batch in iterator:
        batch = batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(optimizer is not None):
            outputs = model(batch)
            loss, mae, rmse = criterion(outputs, batch)
            if optimizer is not None:
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = int(batch.num_graphs)
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["mae"] += float(mae.detach().cpu()) * batch_size
        totals["rmse"] += float(rmse.detach().cpu()) * batch_size
        totals["samples"] += batch_size
        iterator.set_postfix({"mae": f"{float(mae.detach().cpu()):.4f}"})

    samples = max(1, totals["samples"])
    return {key: totals[key] / samples for key in ("loss", "mae", "rmse")}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="/home/yjiao/QH9Stable.db")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-start-lr", type=float, default=1e-7)
    parser.add_argument("--warmup-epochs", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--bottle-hidden-size", type=int, default=64)
    parser.add_argument("--num-gnn-layers", type=int, default=4)
    parser.add_argument("--sh-lmax", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--csa-top-k", type=int, default=8)
    parser.add_argument("--hca-top-k", type=int, default=8)
    parser.add_argument("--hca-lmax", type=int, default=3)
    parser.add_argument("--indexer-compress-dim", type=int, default=32)
    parser.add_argument("--max-radius", type=float, default=7.0)
    parser.add_argument("--radius-embed-dim", type=int, default=64)
    parser.add_argument("--cache-graphs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--graph-cache-path", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="./runs/qhformer_qh9_5k")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"qh9_5k_{timestamp}"
    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(run_dir / "training.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("qhformer_qh9")
    with open(run_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    logger.info("Graph/scatter backend: %s", TORCH_OPS_BACKEND)
    logger.info("Loading QH9 dataset from %s", args.db_path)
    graph_cache_path = args.graph_cache_path
    if args.cache_graphs and not graph_cache_path:
        radius_tag = str(args.max_radius).replace(".", "p")
        graph_cache_path = str(Path("dataset") / f"qh9_graph_cache_{args.max_samples}_r{radius_tag}_l{args.sh_lmax}.pt")
    dataset = QH9SQLiteHamiltonianDataset(
        args.db_path,
        max_samples=args.max_samples,
        cache_graphs=args.cache_graphs,
        graph_cache_path=graph_cache_path,
        max_radius=args.max_radius,
        sh_lmax=args.sh_lmax,
    )
    train_indices, val_indices = split_indices(len(dataset), args.train_split, args.seed)
    dim_counts = defaultdict(int)
    for dim in dataset.orbital_dims:
        dim_counts[int(dim)] += 1
    logger.info("QH9 total rows: %d | selected: %d", dataset.total_rows, len(dataset))
    logger.info("Train/val: %d/%d", len(train_indices), len(val_indices))
    logger.info("Orbital dim buckets: %s", dict(sorted(dim_counts.items())))
    logger.info("Graph cache: %s", graph_cache_path if args.cache_graphs else "disabled")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = QHformer(
        in_node_features=1,
        sh_lmax=args.sh_lmax,
        hidden_size=args.hidden_size,
        bottle_hidden_size=args.bottle_hidden_size,
        num_gnn_layers=args.num_gnn_layers,
        max_radius=args.max_radius,
        num_nodes=10,
        radius_embed_dim=args.radius_embed_dim,
        attention_temperature=1.0,
        num_heads=args.num_heads,
        use_hybrid_attention=True,
        csa_top_k=args.csa_top_k,
        hca_top_k=args.hca_top_k,
        hca_lmax=args.hca_lmax,
        indexer_compress_dim=args.indexer_compress_dim,
        attention_score_residual_init_std=0.0,
    ).to(device)
    model.set(device)
    logger.info("Model parameters: %d", model.get_number_of_parameters())

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_val_mae = float("inf")
    start = time.time()

    for epoch in range(1, args.num_epochs + 1):
        lr = get_lr(epoch, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        train_loader = make_loader(
            dataset, train_indices, args.batch_size, True, args.seed + epoch, args.num_workers
        )
        val_loader = make_loader(
            dataset, val_indices, args.batch_size, False, args.seed, args.num_workers
        )
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.grad_clip, f"Epoch {epoch}")
        val_metrics = run_epoch(model, val_loader, None, device, args.grad_clip, f"Val {epoch}")

        if epoch % args.log_interval == 0:
            logger.info(
                "Epoch %5d | Train MAE %.6f | Val MAE %.6f | Train RMSE %.6f | Val RMSE %.6f | LR %.2e",
                epoch,
                train_metrics["mae"],
                val_metrics["mae"],
                train_metrics["rmse"],
                val_metrics["rmse"],
                lr,
            )

        is_best = val_metrics["mae"] < best_val_mae
        if is_best:
            best_val_mae = val_metrics["mae"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mae": best_val_mae,
                    "args": vars(args),
                },
                run_dir / "best_checkpoint.pth",
            )
            logger.info("New best model at epoch %d (MAE %.6f)", epoch, best_val_mae)

        if epoch % args.save_interval == 0 or epoch == args.num_epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_mae": best_val_mae,
                    "args": vars(args),
                },
                run_dir / "latest_checkpoint.pth",
            )

    logger.info("Finished in %.1f sec | best Val MAE %.6f", time.time() - start, best_val_mae)


if __name__ == "__main__":
    main()
