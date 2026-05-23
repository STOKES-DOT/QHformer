"""Detailed QH9 GNN-layer profiler.

This script instruments only the 4 CSA/HCA attention layers and reports where
their forward time is spent. It is a profiling utility and does not save model
checkpoints.
"""

import argparse
import functools
import os
import sys
import time
from collections import defaultdict

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


class GNNProfiler:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.extra = defaultdict(float)
        self.handles = []
        self.patches = []

    def add(self, name, seconds):
        self.times[name] += float(seconds)
        self.counts[name] += 1

    def _patch_method(self, obj, method_name, label):
        original = getattr(obj, method_name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            sync(self.device)
            start = time.perf_counter()
            output = original(*args, **kwargs)
            sync(self.device)
            self.add(label, time.perf_counter() - start)
            if method_name == "_select_topk":
                selected = int(output.numel()) if hasattr(output, "numel") else 0
                self.extra[f"{label}.selected_edges"] += selected
            return output

        setattr(obj, method_name, wrapped)
        self.patches.append((obj, method_name, original))

    def _hook_module(self, module, label):
        def pre_hook(mod, _inputs):
            sync(self.device)
            mod._qh_gnn_profile_start = time.perf_counter()

        def post_hook(mod, _inputs, _output):
            sync(self.device)
            start = getattr(mod, "_qh_gnn_profile_start", None)
            if start is not None:
                self.add(label, time.perf_counter() - start)

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def __enter__(self):
        for layer_idx, wrapper in enumerate(self.model.e3_gnn_layer):
            conv = wrapper.conv
            prefix = f"gnn{layer_idx}.{conv.__class__.__name__}"
            self._hook_module(wrapper, f"{prefix}.total_wrapper")
            self._patch_method(conv, "_compute_s0", f"{prefix}._compute_s0")
            self._patch_method(conv, "_project_key_value", f"{prefix}._project_key_value")
            self._patch_method(conv, "_multihead_attention", f"{prefix}._multihead_attention")
            if hasattr(conv, "_select_topk"):
                self._patch_method(conv, "_select_topk", f"{prefix}._select_topk")

            for attr in (
                "norm_gate",
                "linear_node",
                "linear_node_pre",
                "inner_product_s0",
                "indexer",
                "linear_query",
                "tp_key",
                "tp_value",
                "fc_key",
                "fc_value",
                "layer_l0_key",
                "layer_l0_value",
                "query_norm",
                "key_norm",
                "inner_product",
                "attention_score",
                "linear_out",
            ):
                module = getattr(conv, attr, None)
                if module is not None:
                    self._hook_module(module, f"{prefix}.{attr}")
        return self

    def __exit__(self, exc_type, exc, tb):
        for obj, method_name, original in self.patches:
            setattr(obj, method_name, original)
        for handle in self.handles:
            handle.remove()

    def print_summary(self, limit=120):
        grouped = defaultdict(float)
        for name, seconds in self.times.items():
            layer = ".".join(name.split(".")[:2])
            grouped[layer] += seconds
        grand = sum(grouped.values())
        print("\nLayer totals")
        print("------------")
        for layer, seconds in sorted(grouped.items(), key=lambda item: item[1], reverse=True):
            print(f"{layer:36s} {seconds:9.3f}s {100.0 * seconds / max(grand, 1e-12):6.2f}%")

        print("\nDetailed timers")
        print("---------------")
        for name, seconds in sorted(self.times.items(), key=lambda item: item[1], reverse=True)[:limit]:
            count = self.counts[name]
            print(f"{name:58s} {seconds:9.3f}s {count:6d} calls {1000.0 * seconds / max(count, 1):9.3f} ms/call")

        print("\nCSA selected edges")
        print("------------------")
        for name, total_selected in sorted(self.extra.items()):
            count = self.counts.get(name.replace(".selected_edges", ""), 0)
            print(f"{name:58s} avg_selected={total_selected / max(count, 1):.1f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="/home/yjiao/QH9Stable.db")
    parser.add_argument("--graph-cache-path", default="dataset/qh9_graph_cache_5000_r7p0_l4.pt")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-train-batches", type=int, default=64)
    parser.add_argument("--max-val-batches", type=int, default=16)
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
    parser.add_argument("--attention-operator", choices=["tp", "so2"], default="tp")
    parser.add_argument("--max-radius", type=float, default=7.0)
    parser.add_argument("--radius-embed-dim", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


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
        attention_operator=args.attention_operator,
    ).to(device)
    model.set(device)
    return model


def run_profile(args):
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
    train_loader = make_loader(dataset, train_indices, args.batch_size, True, args.seed + 1, 0)
    val_loader = make_loader(dataset, val_indices, args.batch_size, False, args.seed, 0)
    model = build_model(args, device)
    optimizer = optim.AdamW(model.parameters(), lr=get_lr(1, args), weight_decay=args.weight_decay)

    print(
        f"device={device} train_batches={len(train_loader)} val_batches={len(val_loader)} "
        f"max_train_batches={args.max_train_batches} max_val_batches={args.max_val_batches}"
    )

    train_metrics = {"samples": 0, "batches": 0, "mae": 0.0}
    val_metrics = {"samples": 0, "batches": 0, "mae": 0.0}
    wall_start = time.perf_counter()

    with GNNProfiler(model, device) as profiler:
        model.train(True)
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx >= args.max_train_batches:
                break
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            loss, mae, _ = criterion(outputs, batch)
            if torch.isfinite(loss):
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            train_metrics["samples"] += int(batch.num_graphs)
            train_metrics["batches"] += 1
            train_metrics["mae"] += float(mae.detach().cpu()) * int(batch.num_graphs)

        model.train(False)
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= args.max_val_batches:
                    break
                batch = batch.to(device)
                outputs = model(batch)
                _, mae, _ = criterion(outputs, batch)
                val_metrics["samples"] += int(batch.num_graphs)
                val_metrics["batches"] += 1
                val_metrics["mae"] += float(mae.detach().cpu()) * int(batch.num_graphs)

    sync(device)
    wall_seconds = time.perf_counter() - wall_start
    train_metrics["mae"] /= max(1, train_metrics["samples"])
    val_metrics["mae"] /= max(1, val_metrics["samples"])
    print(f"wall_seconds={wall_seconds:.3f}")
    print(f"train_metrics={train_metrics}")
    print(f"val_metrics={val_metrics}")
    profiler.print_summary()


if __name__ == "__main__":
    run_profile(parse_args())
