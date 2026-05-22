"""
Profile one QH9 training epoch with explicit CUDA synchronization.

This is intended for bottleneck diagnosis, not for training. It keeps the same
dataset bucketing and model path as train_qhformer_qh9.py, then reports where a
single epoch spends wall time.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.qhformer import QHformer  # noqa: E402
from training.train_qhformer_qh9 import (  # noqa: E402
    QH9SQLiteHamiltonianDataset,
    criterion,
    get_lr,
    make_loader,
    split_indices,
)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class StageTimer:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, name, seconds):
        self.times[name] += float(seconds)
        self.counts[name] += 1

    def summary(self):
        total = sum(self.times.values())
        rows = []
        for name, seconds in sorted(self.times.items(), key=lambda item: item[1], reverse=True):
            count = self.counts[name]
            rows.append({
                "stage": name,
                "seconds": seconds,
                "percent": 100.0 * seconds / total if total > 0 else 0.0,
                "count": count,
                "ms_per_call": 1000.0 * seconds / max(1, count),
            })
        return {"total_seconds": total, "rows": rows}


class ForwardSectionProfiler:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.handles = []

    def _register(self, module, name):
        def pre_hook(mod, _inputs):
            sync(self.device)
            mod._qh_profile_start = time.perf_counter()

        def post_hook(mod, _inputs, _output):
            sync(self.device)
            start = getattr(mod, "_qh_profile_start", None)
            if start is not None:
                self.times[name] += time.perf_counter() - start
                self.counts[name] += 1

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def __enter__(self):
        original_build_final_matrix = self.model.build_final_matrix

        def timed_build_final_matrix(*args, **kwargs):
            sync(self.device)
            start = time.perf_counter()
            output = original_build_final_matrix(*args, **kwargs)
            sync(self.device)
            self.times["forward.build_final_matrix"] += time.perf_counter() - start
            self.counts["forward.build_final_matrix"] += 1
            return output

        self.model.build_final_matrix = timed_build_final_matrix
        self._original_build_final_matrix = original_build_final_matrix
        self._register(self.model.node_embedding, "forward.node_embedding")
        self._register(self.model.distance_expansion, "forward.distance_expansion")
        for idx, layer in enumerate(self.model.e3_gnn_layer):
            self._register(layer, f"forward.gnn_layer_{idx}")
        for idx, layer in enumerate(self.model.e3_gnn_node_layer):
            self._register(layer, f"forward.node_pair_ii_layer_{idx}")
        for idx, layer in enumerate(self.model.e3_gnn_node_pair_layer):
            self._register(layer, f"forward.node_pair_ij_layer_{idx}")
        self._register(self.model.output_ii, "forward.output_ii")
        self._register(self.model.output_ij, "forward.output_ij")
        for key, module in self.model.fc_ii.items():
            self._register(module, f"forward.fc_ii.{key}")
        for key, module in self.model.fc_ii_bias.items():
            self._register(module, f"forward.fc_ii_bias.{key}")
        for key, module in self.model.fc_ij.items():
            self._register(module, f"forward.fc_ij.{key}")
        for key, module in self.model.fc_ij_bias.items():
            self._register(module, f"forward.fc_ij_bias.{key}")
        for key, module in self.model.expand_ii.items():
            self._register(module, f"forward.expand_ii.{key}")
        for key, module in self.model.expand_ij.items():
            self._register(module, f"forward.expand_ij.{key}")
        return self

    def __exit__(self, exc_type, exc, tb):
        if hasattr(self, "_original_build_final_matrix"):
            self.model.build_final_matrix = self._original_build_final_matrix
        for handle in self.handles:
            handle.remove()

    def summary(self, forward_total):
        rows = []
        for name, seconds in sorted(self.times.items(), key=lambda item: item[1], reverse=True):
            count = self.counts[name]
            rows.append({
                "section": name,
                "seconds": seconds,
                "percent_of_forward": 100.0 * seconds / forward_total if forward_total > 0 else 0.0,
                "count": count,
                "ms_per_call": 1000.0 * seconds / max(1, count),
            })
        accounted = sum(self.times.values())
        rows.append({
            "section": "forward.other_unhooked",
            "seconds": max(0.0, forward_total - accounted),
            "percent_of_forward": 100.0 * max(0.0, forward_total - accounted) / forward_total
            if forward_total > 0 else 0.0,
            "count": 0,
            "ms_per_call": 0.0,
        })
        return rows


def build_model(args, device):
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
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
    return model


def profile_loader(model, loader, optimizer, device, grad_clip, timer, train=True, max_batches=0):
    model.train(train)
    metrics = {"loss": 0.0, "mae": 0.0, "rmse": 0.0, "samples": 0, "batches": 0}
    prefix = "train" if train else "val"
    iterator = iter(loader)

    while True:
        if max_batches and metrics["batches"] >= max_batches:
            break

        t0 = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        timer.add(f"{prefix}.data_load", time.perf_counter() - t0)

        t0 = time.perf_counter()
        batch = batch.to(device)
        sync(device)
        timer.add(f"{prefix}.to_device", time.perf_counter() - t0)

        if train:
            t0 = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            timer.add(f"{prefix}.zero_grad", time.perf_counter() - t0)

        with torch.set_grad_enabled(train):
            t0 = time.perf_counter()
            outputs = model(batch)
            sync(device)
            timer.add(f"{prefix}.forward", time.perf_counter() - t0)

            t0 = time.perf_counter()
            loss, mae, rmse = criterion(outputs, batch)
            sync(device)
            timer.add(f"{prefix}.loss", time.perf_counter() - t0)

            if train and torch.isfinite(loss):
                t0 = time.perf_counter()
                loss.backward()
                sync(device)
                timer.add(f"{prefix}.backward", time.perf_counter() - t0)

                if grad_clip > 0:
                    t0 = time.perf_counter()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    sync(device)
                    timer.add(f"{prefix}.grad_clip", time.perf_counter() - t0)

                t0 = time.perf_counter()
                optimizer.step()
                sync(device)
                timer.add(f"{prefix}.optimizer_step", time.perf_counter() - t0)
            elif train:
                timer.add(f"{prefix}.skipped_nonfinite", 0.0)

        t0 = time.perf_counter()
        batch_size = int(batch.num_graphs)
        metrics["loss"] += float(loss.detach().cpu()) * batch_size
        metrics["mae"] += float(mae.detach().cpu()) * batch_size
        metrics["rmse"] += float(rmse.detach().cpu()) * batch_size
        metrics["samples"] += batch_size
        metrics["batches"] += 1
        timer.add(f"{prefix}.metrics_cpu", time.perf_counter() - t0)

    samples = max(1, metrics["samples"])
    return {
        "loss": metrics["loss"] / samples,
        "mae": metrics["mae"] / samples,
        "rmse": metrics["rmse"] / samples,
        "samples": metrics["samples"],
        "batches": metrics["batches"],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="/home/yjiao/QH9Stable.db")
    parser.add_argument("--graph-cache-path", default="dataset/qh9_graph_cache_5000_r7p0_l4.pt")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-start-lr", type=float, default=1e-7)
    parser.add_argument("--warmup-epochs", type=int, default=500)
    parser.add_argument("--num-epochs", type=int, default=2000)
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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--skip-val", action="store_true")
    parser.add_argument("--output", default="runs/profiles/qh9_epoch_profile.json")
    return parser.parse_args()


def print_rows(title, rows, key="stage", limit=20):
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows[:limit]:
        print(
            f"{row[key]:36s} {row['seconds']:9.3f}s "
            f"{row.get('percent', row.get('percent_of_forward', 0.0)):6.2f}% "
            f"{row['count']:6d} calls {row['ms_per_call']:9.3f} ms/call"
        )


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = QH9SQLiteHamiltonianDataset(
        args.db_path,
        max_samples=args.max_samples,
        cache_graphs=True,
        graph_cache_path=args.graph_cache_path,
        max_radius=args.max_radius,
        sh_lmax=args.sh_lmax,
    )
    train_indices, val_indices = split_indices(len(dataset), args.train_split, args.seed)
    train_loader = make_loader(
        dataset, train_indices, args.batch_size, True, args.seed + args.epoch, args.num_workers
    )
    val_loader = make_loader(
        dataset, val_indices, args.batch_size, False, args.seed, args.num_workers
    )

    model = build_model(args, device)
    optimizer = optim.AdamW(model.parameters(), lr=get_lr(args.epoch, args), weight_decay=args.weight_decay)
    timer = StageTimer()

    print(f"device={device} samples={len(dataset)} train_batches={len(train_loader)} val_batches={len(val_loader)}")
    print(f"cache={args.graph_cache_path}")

    wall_start = time.perf_counter()
    with ForwardSectionProfiler(model, device) as forward_profiler:
        train_metrics = profile_loader(
            model,
            train_loader,
            optimizer,
            device,
            args.grad_clip,
            timer,
            train=True,
            max_batches=args.max_train_batches,
        )
        if args.skip_val:
            val_metrics = None
        else:
            val_metrics = profile_loader(
                model,
                val_loader,
                None,
                device,
                args.grad_clip,
                timer,
                train=False,
                max_batches=args.max_val_batches,
            )
    sync(device)
    wall_seconds = time.perf_counter() - wall_start

    stage_summary = timer.summary()
    forward_total = timer.times.get("train.forward", 0.0) + timer.times.get("val.forward", 0.0)
    forward_rows = forward_profiler.summary(forward_total)
    result = {
        "args": vars(args),
        "wall_seconds": wall_seconds,
        "stage_summary": stage_summary,
        "forward_sections": forward_rows,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nwall_seconds={wall_seconds:.3f}")
    print(f"train_metrics={train_metrics}")
    print(f"val_metrics={val_metrics}")
    print_rows("Stage breakdown", stage_summary["rows"], key="stage")
    print_rows("Forward section breakdown", forward_rows, key="section")
    print(f"\njson={output}")


if __name__ == "__main__":
    main()
