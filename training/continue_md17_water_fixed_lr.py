#!/usr/bin/env python
"""Continue an MD17 QHformer run at a fixed learning rate."""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim

import train_qhformer as base


def config_from_base(**overrides):
    values = {
        key: value
        for key, value in vars(base.Config).items()
        if not key.startswith("_") and not callable(value)
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--extra-epochs", type=int, default=1000)
    parser.add_argument("--fixed-lr", type=float, default=1e-5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--log-dir", default="./runs/qhformer_water_continue_fixed_lr")
    args = parser.parse_args()

    config = config_from_base(
        dataset_name="water",
        device=args.device,
        learning_rate=args.fixed_lr,
        min_lr=args.fixed_lr,
        log_dir=args.log_dir,
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"water_continue_lr{args.fixed_lr:g}_{timestamp}"
    run_dir = Path(config.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(run_dir / "training.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info("Continuing MD17 water fixed-LR run")
    logger.info("Checkpoint: %s", args.checkpoint)
    logger.info("Extra epochs: %d", args.extra_epochs)
    logger.info("Fixed LR: %.3e", args.fixed_lr)
    logger.info("Device: %s", args.device)

    with open(run_dir / "config.json", "w") as handle:
        json.dump(vars(config), handle, indent=2)

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    start_epoch = int(checkpoint.get("epoch", 0))
    end_epoch = start_epoch + args.extra_epochs
    config.num_epochs = end_epoch

    model = base.QHformer(
        in_node_features=config.in_node_features,
        sh_lmax=config.sh_lmax,
        hidden_size=config.hidden_size,
        bottle_hidden_size=config.bottle_hidden_size,
        num_gnn_layers=config.num_gnn_layers,
        max_radius=config.max_radius,
        radius_embed_dim=config.radius_embed_dim,
        attention_temperature=config.attention_temperature,
        num_heads=config.num_heads,
        use_hybrid_attention=config.use_hybrid_attention,
        csa_top_k=config.csa_top_k,
        hca_top_k=None,
        hca_lmax=config.hca_lmax,
        indexer_compress_dim=config.indexer_compress_dim,
        attention_score_residual_init_std=config.attention_score_residual_init_std,
    ).to(config.device)
    model.set(config.device)
    incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if incompatible.missing_keys:
        logger.warning("Missing checkpoint keys initialized from current model defaults: %s", incompatible.missing_keys)
    if incompatible.unexpected_keys:
        logger.warning("Unexpected checkpoint keys ignored: %s", incompatible.unexpected_keys)

    full_dataset = base.MD17_DFT(config.data_root, name=config.dataset_name)
    dataset = full_dataset
    num_train = int(len(dataset) * config.train_split)
    num_test = len(dataset) - num_train
    train_dataset, test_dataset = base.random_split(dataset, [num_train, num_test], seed=config.seed)
    train_loader = base.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = base.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    logger.info("Dataset size: %d; train: %d; val: %d", len(dataset), len(train_dataset), len(test_dataset))

    optimizer = optim.AdamW(model.parameters(), lr=args.fixed_lr, weight_decay=config.weight_decay)
    if "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except ValueError as exc:
            logger.warning("Optimizer state is incompatible with the current model; starting fresh optimizer: %s", exc)
    for group in optimizer.param_groups:
        group["lr"] = args.fixed_lr

    history = checkpoint.get("history") or {
        "train_loss": [],
        "train_mae": [],
        "val_loss": [],
        "val_mae": [],
        "epoch": [],
        "lr": [],
    }
    best_val_mae = float(checkpoint.get("best_val_mae", float("inf")))
    best_epoch = int(checkpoint.get("best_epoch", 0))
    logger.info("Starting from epoch %d; existing best MAE %.8f at epoch %d", start_epoch, best_val_mae, best_epoch)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        for group in optimizer.param_groups:
            group["lr"] = args.fixed_lr

        train_errors = base.train_one_epoch(model, train_loader, optimizer, config.device, epoch, config)
        val_errors = base.validate(model, test_loader, config.device, config)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_errors.get("loss", float("nan")))
        history["train_mae"].append(train_errors.get("hamiltonian_mae", float("nan")))
        history["val_loss"].append(val_errors.get("loss", float("nan")))
        history["val_mae"].append(val_errors.get("hamiltonian_mae", float("nan")))
        history["lr"].append(args.fixed_lr)

        val_mae = val_errors.get("hamiltonian_mae", float("nan"))
        is_best = not np.isnan(val_mae) and val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_epoch = epoch

        if epoch % config.log_interval == 0 or epoch == start_epoch + 1:
            logger.info(
                "Epoch %5d | Train MAE: %.6f | Val MAE: %.6f | LR: %.2e",
                epoch,
                train_errors.get("hamiltonian_mae", float("nan")),
                val_mae,
                args.fixed_lr,
            )

        save_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(config),
            "history": history,
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
        }
        if is_best:
            torch.save(save_payload, run_dir / "best_checkpoint.pth")
            logger.info("New best model at epoch %d (MAE: %.8f)", epoch, best_val_mae)

        if epoch % config.save_interval == 0 or epoch == end_epoch:
            base.visualize_predictions(model, train_loader, test_loader, config.device, run_dir, epoch, logger)
            torch.save(save_payload, run_dir / "latest_checkpoint.pth")
            base.plot_training_curves(history, run_dir, logger, config.data_fraction)

    logger.info("Continuation completed!")
    logger.info("Best Val MAE: %.8f at epoch %d", best_val_mae, best_epoch)
    base.plot_training_curves(history, run_dir, logger, config.data_fraction)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(config),
            "history": history,
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
        },
        run_dir / "final_model.pth",
    )


if __name__ == "__main__":
    main()
