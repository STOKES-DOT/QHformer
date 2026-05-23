#!/usr/bin/env python
"""Micro-benchmark SO(2), CSA, and HCA attention operators."""

import argparse
import json
import time
from types import SimpleNamespace

import torch
from e3nn import o3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.inner_product_attention import (  # noqa: E402
    CompressedSparseAttentionLayer,
    HeavyCompressedAttentionLayer,
    MultiHeadAttentionLayer,
    get_feasible_irrep,
)
from models.so2_ops import SO2EdgeConv  # noqa: E402


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) * 1000.0 / iters


def peak_memory_mb(fn, device):
    if device.type != "cuda":
        return None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    fn()
    synchronize(device)
    return torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)


def make_graph(num_nodes, num_edges, edge_attr_dim, sh_lmax, device):
    dst = torch.arange(num_edges, device=device) % num_nodes
    src = (torch.arange(num_edges, device=device) * 7 + 1) % num_nodes
    src = torch.where(src == dst, (src + 1) % num_nodes, src)
    pos = torch.randn(num_nodes, 3, device=device)
    edge_vec = pos[dst] - pos[src]
    sh_irrep = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
    edge_sh = o3.spherical_harmonics(
        sh_irrep,
        edge_vec[:, [1, 2, 0]],
        normalize=True,
        normalization="component",
    )
    edge_attr = torch.randn(num_edges, edge_attr_dim, device=device) * 0.1
    edge_attr[:, 0] = edge_vec.norm(dim=-1)
    return SimpleNamespace(
        pos=pos,
        edge_index=torch.stack([dst, src], dim=0),
        edge_attr=edge_attr,
        edge_sh=edge_sh,
    )


def benchmark_so2(args, device):
    irreps_in = o3.Irreps(
        f"{args.hidden_size}x0e + {args.hidden_size}x1o + "
        f"{args.hidden_size}x2e + {args.hidden_size}x3o + {args.hidden_size}x4e"
    )
    sh_irrep = o3.Irreps.spherical_harmonics(lmax=args.sh_lmax)
    irreps_out, instructions = get_feasible_irrep(irreps_in, sh_irrep, irreps_in, tp_mode="uvu")
    dense = o3.TensorProduct(
        irreps_in,
        sh_irrep,
        irreps_out,
        instructions,
        shared_weights=False,
        internal_weights=False,
    ).to(device)
    so2 = SO2EdgeConv(irreps_in, irreps_in).to(device)

    x = torch.randn(args.num_edges, irreps_in.dim, device=device)
    edge_vec = torch.randn(args.num_edges, 3, device=device)
    edge_sh = torch.randn(args.num_edges, sh_irrep.dim, device=device)
    dense_weight = torch.randn(args.num_edges, dense.weight_numel, device=device) * 0.01
    so2_weight = torch.randn(args.num_edges, so2.weight_numel, device=device) * 0.01

    dense_ms = time_call(lambda: dense(x, edge_sh, dense_weight), args.warmup, args.iters, device)
    so2_ms = time_call(lambda: so2(x, edge_vec, so2_weight), args.warmup, args.iters, device)
    dense_fn = lambda: dense(x, edge_sh, dense_weight)
    so2_fn = lambda: so2(x, edge_vec, so2_weight)
    return {
        "dense_tp_weight_numel": dense.weight_numel,
        "so2_weight_numel": so2.weight_numel,
        "dense_tp_ms": dense_ms,
        "so2_ms": so2_ms,
        "so2_vs_tp_speedup": dense_ms / so2_ms if so2_ms > 0 else float("inf"),
        "dense_peak_mb": peak_memory_mb(dense_fn, device),
        "so2_peak_mb": peak_memory_mb(so2_fn, device),
    }


def benchmark_attention_layer(args, device):
    irreps = o3.Irreps(
        f"{args.hidden_size}x0e + {args.hidden_size}x1o + "
        f"{args.hidden_size}x2e + {args.hidden_size}x3o + {args.hidden_size}x4e"
    )
    sh_irrep = o3.Irreps.spherical_harmonics(lmax=args.sh_lmax)
    data = make_graph(args.num_nodes, args.num_edges, args.edge_attr_dim, args.sh_lmax, device)
    x = torch.randn(args.num_nodes, irreps.dim, device=device) * 0.1

    def make_layer(**kwargs):
        layer = MultiHeadAttentionLayer(
            irrep_in_node=irreps,
            irrep_hidden=irreps,
            irrep_out=irreps,
            sh_irrep=sh_irrep,
            edge_attr_dim=args.edge_attr_dim,
            node_attr_dim=args.hidden_size,
            invariant_layers=1,
            invariant_neurons=args.invariant_neurons,
            use_norm_gate=True,
            num_heads=args.num_heads,
            **kwargs,
        ).to(device)
        layer.eval()
        return layer

    dense = make_layer(attention_operator="tp")
    so2 = make_layer(attention_operator="so2")

    with torch.no_grad():
        dense_fn = lambda: dense(data, x)
        so2_fn = lambda: so2(data, x)
        dense_ms = time_call(dense_fn, args.warmup, args.iters, device)
        so2_ms = time_call(so2_fn, args.warmup, args.iters, device)

    return {
        "dense_attention_ms": dense_ms,
        "so2_attention_ms": so2_ms,
        "so2_vs_dense_speedup": dense_ms / so2_ms if so2_ms > 0 else float("inf"),
        "dense_attention_peak_mb": peak_memory_mb(dense_fn, device),
        "so2_attention_peak_mb": peak_memory_mb(so2_fn, device),
    }


def _so2_cache_stats(layer):
    modules = [m for m in layer.modules() if isinstance(m, SO2EdgeConv)]
    if not modules:
        return None
    return {
        "hits": sum(m.rotation_cache_hits for m in modules),
        "misses": sum(m.rotation_cache_misses for m in modules),
        "entries": sum(m.rotation_cache_size for m in modules),
    }


def benchmark_combined_attention(args, device):
    irreps = o3.Irreps(
        f"{args.hidden_size}x0e + {args.hidden_size}x1o + "
        f"{args.hidden_size}x2e + {args.hidden_size}x3o + {args.hidden_size}x4e"
    )
    sh_irrep = o3.Irreps.spherical_harmonics(lmax=args.sh_lmax)
    data = make_graph(args.num_nodes, args.num_edges, args.edge_attr_dim, args.sh_lmax, device)
    x = torch.randn(args.num_nodes, irreps.dim, device=device) * 0.1

    base_kwargs = dict(
        irrep_in_node=irreps,
        irrep_hidden=irreps,
        irrep_out=irreps,
        sh_irrep=sh_irrep,
        edge_attr_dim=args.edge_attr_dim,
        node_attr_dim=args.hidden_size,
        invariant_layers=1,
        invariant_neurons=args.invariant_neurons,
        use_norm_gate=True,
        num_heads=args.num_heads,
    )

    dense_tp = MultiHeadAttentionLayer(**base_kwargs, attention_operator="tp").to(device).eval()
    csa_tp = CompressedSparseAttentionLayer(
        **base_kwargs,
        attention_operator="tp",
        top_k=args.top_k,
        indexer_compress_dim=args.indexer_compress_dim,
    ).to(device).eval()
    csa_so2 = CompressedSparseAttentionLayer(
        **base_kwargs,
        attention_operator="so2",
        top_k=args.top_k,
        indexer_compress_dim=args.indexer_compress_dim,
    ).to(device).eval()
    hca_tp = HeavyCompressedAttentionLayer(
        **base_kwargs,
        attention_operator="tp",
        top_k=args.top_k,
        hca_lmax=args.hca_lmax,
        indexer_compress_dim=args.indexer_compress_dim,
    ).to(device).eval()
    hca_so2 = HeavyCompressedAttentionLayer(
        **base_kwargs,
        attention_operator="so2",
        top_k=args.top_k,
        hca_lmax=args.hca_lmax,
        indexer_compress_dim=args.indexer_compress_dim,
    ).to(device).eval()

    layers = {
        "dense_tp": dense_tp,
        "csa_tp": csa_tp,
        "csa_so2": csa_so2,
        "hca_tp": hca_tp,
        "hca_so2": hca_so2,
    }

    timings = {}
    peaks = {}
    with torch.no_grad():
        for name, layer in layers.items():
            fn = lambda layer=layer: layer(data, x)
            timings[f"{name}_ms"] = time_call(fn, args.warmup, args.iters, device)
            peaks[f"{name}_peak_mb"] = peak_memory_mb(fn, device)

    dense_ms = timings["dense_tp_ms"]
    dense_peak = peaks["dense_tp_peak_mb"]
    selected_edges_upper_bound = min(args.top_k * args.num_nodes, args.num_edges)
    results = {
        "top_k": args.top_k,
        "hca_lmax": args.hca_lmax,
        "num_edges": args.num_edges,
        "selected_edges_upper_bound": selected_edges_upper_bound,
    }
    for name, layer in layers.items():
        ms = timings[f"{name}_ms"]
        peak = peaks[f"{name}_peak_mb"]
        results[f"{name}_ms"] = ms
        results[f"{name}_vs_dense_speedup"] = dense_ms / ms if ms > 0 else float("inf")
        results[f"{name}_peak_mb"] = peak
        if peak is not None and dense_peak is not None:
            results[f"{name}_memory_ratio"] = peak / dense_peak
        cache_stats = _so2_cache_stats(layer)
        if cache_stats is not None:
            results[f"{name}_so2_cache"] = cache_stats
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-nodes", type=int, default=32)
    parser.add_argument("--num-edges", type=int, default=512)
    parser.add_argument("--sh-lmax", type=int, default=4)
    parser.add_argument("--edge-attr-dim", type=int, default=16)
    parser.add_argument("--invariant-neurons", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hca-lmax", type=int, default=3)
    parser.add_argument("--indexer-compress-dim", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(123)
    results = {
        "device": str(device),
        "hidden_size": args.hidden_size,
        "num_edges": args.num_edges,
        "so2_operator": benchmark_so2(args, device),
        "attention_layer": benchmark_attention_layer(args, device),
        "combined_attention": benchmark_combined_attention(args, device),
    }
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for section, payload in results.items():
            if isinstance(payload, dict):
                print(f"[{section}]")
                for key, value in payload.items():
                    print(f"{key}: {value}")
            else:
                print(f"{section}: {payload}")


if __name__ == "__main__":
    main()
