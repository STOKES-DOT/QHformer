#!/usr/bin/env python
"""Train QHformer v2 on a fixed-size MD17 molecule subset."""

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
    parser.add_argument("--molecule", default="uracil")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--num-epochs", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-epochs", type=int, default=1000)
    parser.add_argument("--warmup-start-lr", type=float, default=1e-7)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log-dir", default="./runs/qhformer_md17_subset")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = config_from_base(
        dataset_name=args.molecule,
        data_fraction=1.0,
        train_split=args.train_split,
        test_split=1.0 - args.train_split,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr,
        warmup_epochs=args.warmup_epochs,
        warmup_start_lr=args.warmup_start_lr,
        device=args.device,
        log_dir=args.log_dir,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        seed=args.seed,
    )

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"{args.molecule}{args.max_samples}_h{config.hidden_size}_g{config.num_gnn_layers}_{timestamp}"
    run_dir = Path(config.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(run_dir / "training.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info("QHformer v2 MD17 subset training")
    logger.info("Molecule: %s", args.molecule)
    logger.info("Max samples: %d", args.max_samples)
    logger.info("Device: %s", args.device)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Epochs: %d", args.num_epochs)
    logger.info("LR schedule: warmup %.2e -> %.2e, min %.2e", args.warmup_start_lr, args.learning_rate, args.min_lr)

    with open(run_dir / "config.json", "w") as handle:
        payload = vars(config).copy()
        payload["max_samples"] = args.max_samples
        json.dump(payload, handle, indent=2)

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
        hca_top_k=config.hca_top_k,
        hca_lmax=config.hca_lmax,
        indexer_compress_dim=config.indexer_compress_dim,
        attention_score_residual_init_std=config.attention_score_residual_init_std,
    ).to(config.device)
    model.set(config.device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    full_dataset = base.MD17_DFT(config.data_root, name=args.molecule)
    if args.max_samples > 0 and args.max_samples < len(full_dataset):
        generator = torch.Generator().manual_seed(config.seed)
        indices = torch.randperm(len(full_dataset), generator=generator)[: args.max_samples]
        dataset = torch.utils.data.Subset(full_dataset, indices)
    else:
        dataset = full_dataset

    num_train = int(len(dataset) * config.train_split)
    num_test = len(dataset) - num_train
    train_dataset, test_dataset = base.random_split(dataset, [num_train, num_test], seed=config.seed)
    train_loader = base.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = base.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    logger.info("Dataset size: %d; train: %d; val: %d", len(dataset), len(train_dataset), len(test_dataset))

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history = {"train_loss": [], "train_mae": [], "val_loss": [], "val_mae": [], "epoch": [], "lr": []}
    best_val_mae = float("inf")
    best_epoch = 0

    for epoch in range(1, config.num_epochs + 1):
        lr = base.get_lr(epoch, config)
        for group in optimizer.param_groups:
            group["lr"] = lr

        train_errors = base.train_one_epoch(model, train_loader, optimizer, config.device, epoch, config)
        val_errors = base.validate(model, test_loader, config.device, config)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_errors.get("loss", float("nan")))
        history["train_mae"].append(train_errors.get("hamiltonian_mae", float("nan")))
        history["val_loss"].append(val_errors.get("loss", float("nan")))
        history["val_mae"].append(val_errors.get("hamiltonian_mae", float("nan")))
        history["lr"].append(lr)

        val_mae = val_errors.get("hamiltonian_mae", float("nan"))
        is_best = not np.isnan(val_mae) and val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            best_epoch = epoch

        if epoch % config.log_interval == 0 or epoch == 1:
            logger.info(
                "Epoch %5d | Train MAE: %.6f | Val MAE: %.6f | LR: %.2e",
                epoch,
                train_errors.get("hamiltonian_mae", float("nan")),
                val_mae,
                lr,
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": vars(config),
            "history": history,
            "best_val_mae": best_val_mae,
            "best_epoch": best_epoch,
        }
        if is_best:
            torch.save(checkpoint, run_dir / "best_checkpoint.pth")
            logger.info("New best model at epoch %d (MAE: %.8f)", epoch, best_val_mae)

        if epoch % config.save_interval == 0 or epoch == config.num_epochs:
            base.visualize_predictions(model, train_loader, test_loader, config.device, run_dir, epoch, logger)
            torch.save(checkpoint, run_dir / "latest_checkpoint.pth")
            base.plot_training_curves(history, run_dir, logger, config.data_fraction)

    logger.info("Training completed!")
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
