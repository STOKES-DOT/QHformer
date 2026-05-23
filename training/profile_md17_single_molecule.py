#!/usr/bin/env python
"""Profile repeated training steps on one MD17/SchNOrb molecule."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.measure_water_memory import load_samples, make_model, mib  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/home/yjiao/QHformer/dataset")
    parser.add_argument("--molecule", default="uracil")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--bottle-hidden-size", type=int, default=64)
    parser.add_argument("--num-gnn-layers", type=int, default=4)
    parser.add_argument("--radius-embed-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--csa-top-k", type=int, default=8)
    parser.add_argument("--hca-lmax", type=int, default=3)
    parser.add_argument("--indexer-compress-dim", type=int, default=32)
    parser.add_argument("--attention-score-residual-init-std", type=float, default=0.0)
    parser.add_argument("--attention-operator", choices=["tp", "so2"], default="so2")
    parser.add_argument("--optimizer-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=500)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    return parser.parse_args()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def symmetrized_loss(output, target):
    target = target.view(output.shape[0], output.shape[1], output.shape[2])
    pred = 0.5 * (output + output.transpose(-1, -2))
    target = 0.5 * (target + target.transpose(-1, -2))
    diff = pred - target
    mae = diff.abs().mean()
    mse = diff.square().mean()
    loss = mae + mse
    rmse = torch.sqrt(mse)
    return loss, mae, rmse


def train_step(model, optimizer, batch, grad_clip):
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)["hamiltonian"]
    loss, mae, rmse = symmetrized_loss(output, batch.hamiltonian)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {float(loss.detach().cpu())}")
    loss.backward()
    if grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return {
        "loss": float(loss.detach().cpu()),
        "mae": float(mae.detach().cpu()),
        "rmse": float(rmse.detach().cpu()),
    }


def memory_snapshot(device):
    if device.type != "cuda":
        return {}
    return {
        "allocated_mib": mib(torch.cuda.memory_allocated(device)),
        "reserved_mib": mib(torch.cuda.memory_reserved(device)),
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved(device)),
    }


def write_csv(path, rows):
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "elapsed_s",
        "avg_step_ms",
        "loss",
        "mae",
        "rmse",
        "allocated_mib",
        "reserved_mib",
        "peak_allocated_mib",
        "peak_reserved_mib",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    samples = load_samples(args.data_root, args.molecule, args.sample_index + 1)
    sample = samples[args.sample_index]
    batch = Batch.from_data_list([sample]).to(device)

    model = make_model(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.optimizer_lr, weight_decay=args.weight_decay)
    parameter_count = sum(param.numel() for param in model.parameters() if param.requires_grad)

    synchronize(device)
    model_memory = memory_snapshot(device)

    for _ in range(args.warmup_steps):
        train_step(model, optimizer, batch, args.grad_clip)
    synchronize(device)
    post_warmup_memory = memory_snapshot(device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)

    rows = []
    start = time.perf_counter()
    last_metrics = {}
    for epoch in range(1, args.epochs + 1):
        last_metrics = train_step(model, optimizer, batch, args.grad_clip)
        if epoch % args.log_interval == 0 or epoch == 1 or epoch == args.epochs:
            synchronize(device)
            elapsed = time.perf_counter() - start
            row = {
                "epoch": epoch,
                "elapsed_s": elapsed,
                "avg_step_ms": elapsed * 1000.0 / epoch,
                **last_metrics,
                **memory_snapshot(device),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    synchronize(device)
    total_elapsed = time.perf_counter() - start
    final_memory = memory_snapshot(device)
    summary = {
        "molecule": args.molecule,
        "sample_index": args.sample_index,
        "epochs": args.epochs,
        "warmup_steps": args.warmup_steps,
        "device": str(device),
        "attention_operator": args.attention_operator,
        "hidden_size": args.hidden_size,
        "bottle_hidden_size": args.bottle_hidden_size,
        "num_gnn_layers": args.num_gnn_layers,
        "hca_lmax": args.hca_lmax,
        "num_atoms": int(batch.atoms.shape[0]),
        "hamiltonian_shape": list(batch.hamiltonian.shape),
        "parameter_count": int(parameter_count),
        "total_elapsed_s": total_elapsed,
        "avg_step_ms": total_elapsed * 1000.0 / args.epochs,
        "final_loss": last_metrics.get("loss"),
        "final_mae": last_metrics.get("mae"),
        "final_rmse": last_metrics.get("rmse"),
        "model_memory": model_memory,
        "post_warmup_memory": post_warmup_memory,
        "timed_peak_memory": final_memory,
    }

    print(json.dumps({"summary": summary}, indent=2, sort_keys=True))
    write_csv(args.output_csv, rows)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as handle:
            json.dump({"summary": summary, "log": rows}, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
