"""
QHformer Training Script

Trains the QHformer model (QHNet with multi-head hybrid CSA/HCA attention) for
predicting quantum Hamiltonian matrices.

Key Features:
- Multi-head CSA/HCA attention: Query/Key maintain equivariant irreps
- Rotation-invariant attention via InnerProduct
- Full equivariance preserved

Usage:
    python train_qhformer.py
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
# Compatibility layers for environments without torch_cluster/torch_scatter
# =============================================================================

def radius_graph_pure(pos, r, batch=None, loop=False, max_num_neighbors=32, flow='source_to_target'):
    """Pure PyTorch radius graph implementation"""
    num_nodes = pos.shape[0]
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=pos.device)

    batch_size = int(batch.max().item()) + 1
    edge_indices = []

    for b in range(batch_size):
        mask = batch == b
        batch_indices = torch.nonzero(mask).squeeze(-1)
        batch_pos = pos[batch_indices]
        num_batch_nodes = batch_pos.shape[0]

        if num_batch_nodes == 0:
            continue

        diff = batch_pos.unsqueeze(1) - batch_pos.unsqueeze(0)
        dist_mat = torch.norm(diff, dim=-1)
        adj = dist_mat <= r
        if not loop:
            adj.fill_diagonal_(False)

        edge_index = adj.nonzero().t()
        if edge_index.numel() > 0:
            edge_index[0] = batch_indices[edge_index[0]]
            edge_index[1] = batch_indices[edge_index[1]]
            edge_indices.append(edge_index)

    if edge_indices:
        return torch.cat(edge_indices, dim=1)
    else:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)


def scatter_pure(src, index, dim_size=None, dim=0, out=None, reduce='sum', dim_size_is_batch=False):
    """Simple scatter implementation"""
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    if index.dim() == 1:
        index_shape = [1] * src.dim()
        index_shape[dim] = index.shape[0]
        index_expanded = index.reshape(index_shape).expand_as(src)
    else:
        index_expanded = index

    if reduce == 'sum' or reduce == 'add':
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index_expanded, src)
    elif reduce == 'mean':
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        counts = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        out.scatter_add_(dim, index_expanded, src)
        counts.scatter_add_(dim, index_expanded, torch.ones_like(src))
        return out / (counts.clamp(min=1))
    else:
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index_expanded, src)


class FakeTorchCluster:
    def __init__(self):
        self.radius_graph = radius_graph_pure


class FakeTorchScatter:
    def __init__(self):
        self.scatter = scatter_pure
        self.scatter_add = scatter_pure
        self.scatter_mul = scatter_pure


# Replace modules BEFORE imports
sys.modules['torch_cluster'] = FakeTorchCluster()
sys.modules['torch_scatter'] = FakeTorchScatter()

# Now import dataset and model
try:
    from utils.ori_dataset import MD17_DFT, random_split
except ImportError:
    # Fallback if utils not available
    print("Warning: utils.ori_dataset not found, using placeholder")
    MD17_DFT = None
    random_split = None

from models.qhformer import QHformer


class Config:
    """QHformer Configuration - Full Dataset Training"""

    # Model architecture
    in_node_features = 4
    sh_lmax = 4
    hidden_size = 256
    bottle_hidden_size = 64
    num_gnn_layers = 4  # 4 GNN layers
    max_radius = 12
    radius_embed_dim = 64

    # Hybrid attention parameters
    attention_temperature = 1.0  # Temperature for attention softmax
    num_heads = 4
    use_hybrid_attention = True
    csa_top_k = 8
    hca_lmax = 3
    indexer_compress_dim = 32
    attention_score_residual_init_std = 0.0

    # Training parameters
    num_epochs = 15000
    batch_size = 512
    learning_rate = 1e-3
    weight_decay = 1e-4
    grad_clip = 0.5

    # Warmup and cosine annealing
    warmup_epochs = 1000  # Match QHTransformer
    warmup_start_lr = 1e-7  # Match QHTransformer
    min_lr = 1e-5
    lr_drop_epoch = None
    lr_drop_gamma = 0.1

    # Dataset
    dataset_name = 'water'
    data_fraction = 1.0  # Modified: full dataset
    train_split = 0.8
    test_split = 0.2

    # Device
    device = 'cuda:0'
    seed = 42

    # Logging
    save_interval = 50  # Visualize and checkpoint every 50 epochs
    log_interval = 10
    log_dir = './runs/qhformer_water_full'

    # Paths - server path
    data_root = '/home/yjiao/QHformer/dataset'

    # Loss weights
    loss_weights = {'hamiltonian': 1.0}


def criterion(outputs, target, loss_weights={'hamiltonian': 1.0}, batch_size=None):
    """Compute loss"""
    error_dict = {}

    for key in loss_weights.keys():
        if key not in outputs or key not in target:
            continue

        H_pred = outputs[key]
        H_true = target[key]

        # Reshape H_true from [batch*n_orb, n_orb] to [batch, n_orb, n_orb]
        if H_true.dim() == 2 and H_pred.dim() == 3:
            n_orbitals = H_pred.shape[1]
            if batch_size is None:
                batch_size = H_true.shape[0] // n_orbitals
            H_true = H_true.view(batch_size, n_orbitals, n_orbitals)

        # Ensure symmetric
        if H_pred.dim() == 3:
            for i in range(H_pred.shape[0]):
                H_pred[i] = (H_pred[i] + H_pred[i].transpose(-1, -2)) / 2.0
        if H_true.dim() == 3:
            for i in range(H_true.shape[0]):
                H_true[i] = (H_true[i] + H_true[i].transpose(-1, -2)) / 2.0

        diff = H_pred - H_true
        mse = torch.mean(diff**2)
        mae = torch.mean(torch.abs(diff))

        error_dict[f'{key}_mae'] = mae
        error_dict[f'{key}_rmse'] = torch.sqrt(mse)
        loss = mse + mae

        if 'loss' in error_dict:
            error_dict['loss'] = error_dict['loss'] + loss_weights[key] * loss
        else:
            error_dict['loss'] = loss_weights[key] * loss

    return error_dict


def get_lr(epoch, config):
    """Learning rate with warmup and cosine annealing - Match E3-DeepMolH"""
    if epoch <= config.warmup_epochs:
        # Linear warmup from warmup_start_lr to learning_rate
        lr = config.warmup_start_lr + (config.learning_rate - config.warmup_start_lr) * epoch / config.warmup_epochs
    else:
        # Cosine annealing from learning_rate to min_lr
        progress = (epoch - config.warmup_epochs) / (config.num_epochs - config.warmup_epochs)
        lr = config.min_lr + (config.learning_rate - config.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

    if config.lr_drop_epoch is not None and epoch > config.lr_drop_epoch:
        lr *= config.lr_drop_gamma
    return lr


def plot_training_curves(history, log_dir, logger, data_fraction=0.01):
    """Plot training curves with log scale - matching QHNet style"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.ticker as mticker

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Progress - {data_fraction*100:.0f}% Data',
                 fontsize=16, fontweight='bold')

    epochs = history['epoch']

    # Loss plot (log scale)
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], label='Train Loss', alpha=0.7)
    ax.plot(epochs, history['val_loss'], label='Val Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curve (Log Scale)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))

    # MAE plot (log scale)
    ax = axes[0, 1]
    ax.plot(epochs, history['train_mae'], label='Train MAE', alpha=0.7)
    ax.plot(epochs, history['val_mae'], label='Val MAE', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (Hartree)')
    ax.set_title('MAE Curve (Log Scale)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))

    # Learning rate plot (log scale)
    ax = axes[1, 0]
    ax.plot(epochs, history['lr'], color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule (Log Scale)', fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))

    # Zoom on last 1000 epochs (log scale)
    ax = axes[1, 1]
    if len(epochs) > 1000:
        zoom_start = max(0, len(epochs) - 1000)
        zoom_epochs = epochs[zoom_start:]
        zoom_train_mae = history['train_mae'][zoom_start:]
        zoom_val_mae = history['val_mae'][zoom_start:]

        ax.plot(zoom_epochs, zoom_train_mae, label='Train MAE', alpha=0.7)
        ax.plot(zoom_epochs, zoom_val_mae, label='Val MAE', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE (Hartree)')
        ax.set_title('Last 1000 Epochs (Zoom, Log Scale)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8))
    else:
        ax.text(0.5, 0.5, 'Not enough data yet', ha='center', va='center')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(log_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("Training curves updated")


def collect_predictions(model, dataloader, device, max_samples=4):
    """Collect predictions for visualization"""
    model.eval()
    predictions = []
    ground_truths = []

    collected = 0
    with torch.no_grad():
        for batch in dataloader:
            if collected >= max_samples:
                break

            batch = batch.to(device)

            # Convert to float32
            if hasattr(batch, 'hamiltonian') and batch.hamiltonian.dtype == torch.float64:
                batch.hamiltonian = batch.hamiltonian.float()
            if hasattr(batch, 'pos') and batch.pos.dtype == torch.float64:
                batch.pos = batch.pos.float()

            outputs = model(batch)
            H_pred = outputs['hamiltonian']
            H_true = batch.hamiltonian

            # Reshape H_true if needed
            if H_true.dim() == 2 and H_pred.dim() == 3:
                n_orbitals = H_pred.shape[1]
                batch_size = H_true.shape[0] // n_orbitals
                H_true = H_true.view(batch_size, n_orbitals, n_orbitals)

            # Collect individual samples from this batch
            num_in_batch = H_pred.shape[0]
            for i in range(num_in_batch):
                if collected >= max_samples:
                    break
                predictions.append(H_pred[i].cpu().numpy())
                ground_truths.append(H_true[i].cpu().numpy())
                collected += 1

    return predictions, ground_truths


def visualize_predictions(model, train_loader, val_loader, device, log_dir, epoch, logger):
    """Visualize predictions for train and val sets"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Collect predictions
    train_preds, train_gt = collect_predictions(model, train_loader, device, max_samples=4)
    val_preds, val_gt = collect_predictions(model, val_loader, device, max_samples=4)

    # Plot predictions
    _plot_predictions(train_preds, train_gt, 'train', log_dir)
    _plot_predictions(val_preds, val_gt, 'val', log_dir)
    _plot_error_distribution(train_preds, train_gt, val_preds, val_gt, log_dir)

    logger.info(f"Visualization updated (epoch {epoch})")


def _plot_predictions(predictions, ground_truths, split, log_dir):
    """Plot predictions for a split"""
    num_samples = len(predictions)
    if num_samples == 0:
        return

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i, (H_pred, H_true) in enumerate(zip(predictions, ground_truths)):
        # Ground truth
        im0 = axes[i, 0].imshow(H_true, cmap='RdBu_r', aspect='auto')
        axes[i, 0].set_title(f'Sample {i+1} - Ground Truth', fontweight='bold')
        plt.colorbar(im0, ax=axes[i, 0])

        # Prediction
        im1 = axes[i, 1].imshow(H_pred, cmap='RdBu_r', aspect='auto')
        axes[i, 1].set_title(f'Sample {i+1} - Prediction', fontweight='bold')
        plt.colorbar(im1, ax=axes[i, 1])

        # Absolute difference
        abs_diff = np.abs(H_pred - H_true)
        im2 = axes[i, 2].imshow(abs_diff, cmap='hot', aspect='auto')
        axes[i, 2].set_title(f'Sample {i+1} - Abs Error (max={abs_diff.max():.4f})', fontweight='bold')
        plt.colorbar(im2, ax=axes[i, 2])

    fig.suptitle(f'QHformer - {split.capitalize()} Set Predictions',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(log_dir / f'predictions_{split}.png', dpi=300, bbox_inches='tight')
    plt.close()


def _plot_error_distribution(train_preds, train_gt, val_preds, val_gt, log_dir):
    """Plot error distribution for train and val"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('QHNet - Error Distribution',
                 fontsize=16, fontweight='bold')

    # Collect all errors
    def get_errors(preds, gts):
        all_abs = []
        diagonal_abs = []
        off_diagonal_abs = []
        for H_pred, H_true in zip(preds, gts):
            abs_diff = np.abs(H_pred - H_true)
            all_abs.extend(abs_diff.flatten())

            n = abs_diff.shape[0]
            diagonal_abs.extend(np.diag(abs_diff))
            off_mask = ~np.eye(n, dtype=bool)
            off_diagonal_abs.extend(abs_diff[off_mask])
        return all_abs, diagonal_abs, off_diagonal_abs

    train_abs, train_diag, train_off = get_errors(train_preds, train_gt)
    val_abs, val_diag, val_off = get_errors(val_preds, val_gt)

    # Train - Absolute error histogram
    ax = axes[0, 0]
    ax.hist(train_abs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(train_abs), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(train_abs):.4f}')
    ax.axvline(np.median(train_abs), color='green', linestyle='--', linewidth=2,
              label=f'Median: {np.median(train_abs):.4f}')
    ax.set_xlabel('Absolute Error (Hartree)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Train Set - Absolute Error', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Val - Absolute error histogram
    ax = axes[0, 1]
    ax.hist(val_abs, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(val_abs), color='red', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(val_abs):.4f}')
    ax.axvline(np.median(val_abs), color='green', linestyle='--', linewidth=2,
              label=f'Median: {np.median(val_abs):.4f}')
    ax.set_xlabel('Absolute Error (Hartree)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Val Set - Absolute Error', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Diagonal vs off-diagonal comparison
    ax = axes[1, 0]
    positions = [1, 2, 3, 4]
    bp = ax.boxplot([train_diag, train_off, val_diag, val_off], positions=positions,
                   widths=0.6, patch_artist=True,
                   labels=['Train-Diag', 'Train-Off', 'Val-Diag', 'Val-Off'])
    colors = ['lightblue', 'steelblue', 'lightcoral', 'coral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_ylabel('Absolute Error (Hartree)', fontsize=11)
    ax.set_title('Diagonal vs Off-Diagonal Errors', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Cumulative distribution
    ax = axes[1, 1]
    train_sorted = np.sort(train_abs)
    val_sorted = np.sort(val_abs)
    train_cum = np.arange(1, len(train_sorted) + 1) / len(train_sorted) * 100
    val_cum = np.arange(1, len(val_sorted) + 1) / len(val_sorted) * 100
    ax.plot(train_sorted, train_cum, linewidth=2, color='steelblue', label='Train')
    ax.plot(val_sorted, val_cum, linewidth=2, color='coral', label='Val')
    ax.set_xlabel('Absolute Error (Hartree)', fontsize=11)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=11)
    ax.set_title('Cumulative Error Distribution', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(log_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def train_one_epoch(model, dataloader, optimizer, device, epoch, config):
    model.train()
    total_error_dict = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{config.num_epochs}')

    for batch in dataloader:
        batch = batch.to(device)

        if batch.pos.dtype == torch.float64:
            batch.pos = batch.pos.float()
        if hasattr(batch, 'hamiltonian') and batch.hamiltonian.dtype == torch.float64:
            batch.hamiltonian = batch.hamiltonian.float()

        optimizer.zero_grad()
        outputs = model(batch)
        target = {'hamiltonian': batch.hamiltonian}

        error_dict = criterion(outputs, target, config.loss_weights, batch_size=batch.num_graphs)
        loss = error_dict['loss']

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN/Inf loss at epoch {epoch}, skipping batch")
            continue

        loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

        optimizer.step()

        for key, value in error_dict.items():
            if key not in total_error_dict:
                total_error_dict[key] = 0.0
            total_error_dict[key] += value.item() if torch.is_tensor(value) else value

        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Handle case where all batches had NaN loss
    if num_batches == 0:
        print(f"Warning: All batches had NaN/Inf loss at epoch {epoch}")
        return {'loss': float('nan'), 'hamiltonian_mae': float('nan'), 'hamiltonian_rmse': float('nan')}

    for key in total_error_dict:
        total_error_dict[key] /= num_batches

    return total_error_dict


def validate(model, dataloader, device, config):
    model.eval()
    total_error_dict = {}
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            if batch.pos.dtype == torch.float64:
                batch.pos = batch.pos.float()
            if hasattr(batch, 'hamiltonian') and batch.hamiltonian.dtype == torch.float64:
                batch.hamiltonian = batch.hamiltonian.float()

            outputs = model(batch)
            target = {'hamiltonian': batch.hamiltonian}
            error_dict = criterion(outputs, target, config.loss_weights, batch_size=batch.num_graphs)

            batch_size = batch.num_graphs
            for key, value in error_dict.items():
                if key not in total_error_dict:
                    total_error_dict[key] = 0.0
                total_error_dict[key] += value.item() * batch_size if torch.is_tensor(value) else value * batch_size
            total_samples += batch_size

    for key in total_error_dict:
        total_error_dict[key] /= total_samples

    return total_error_dict


def main():
    config = Config()

    # Set seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path(f'{config.log_dir}/{timestamp}')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("QHformer Training")
    logger.info("(QHNet with Inner Product Attention)")
    logger.info("=" * 80)
    logger.info(f"Device: {config.device}")
    logger.info(f"Dataset: {config.dataset_name}")
    logger.info(f"Data fraction: {config.data_fraction*100:.1f}%")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Max LR: {config.learning_rate}")
    logger.info(f"Attention temperature: {config.attention_temperature}")
    logger.info("")
    logger.info("Key Innovation:")
    logger.info("  - Query/Key maintain FULL irreps (no scalar compression)")
    logger.info("  - InnerProduct couples irreps to scalars for attention")
    logger.info("  - Value preserves complete equivariant information")

    # Save config
    with open(log_dir / 'config.json', 'w') as f:
        json.dump(vars(config), f, indent=2)

    # Create model
    logger.info("")
    logger.info("Creating model...")
    model = QHformer(
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
        hca_lmax=config.hca_lmax,
        indexer_compress_dim=config.indexer_compress_dim,
        attention_score_residual_init_std=config.attention_score_residual_init_std,
    ).to(config.device)

    # Set device for model-specific tensors
    model.set(config.device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Load dataset
    if MD17_DFT is None:
        logger.error("Dataset not available. Please provide ori_dataset.py")
        return

    logger.info("")
    logger.info("Loading dataset...")
    full_dataset = MD17_DFT(config.data_root, name=config.dataset_name)

    # Use fraction of dataset for testing
    if hasattr(config, 'data_fraction') and config.data_fraction < 1.0:
        num_samples = int(len(full_dataset) * config.data_fraction)
        indices = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(config.seed))[:num_samples]
        dataset = torch.utils.data.Subset(full_dataset, indices)
        logger.info(f"Using {config.data_fraction*100:.1f}% of dataset: {len(dataset)} samples")
    else:
        dataset = full_dataset

    num_train = int(len(dataset) * config.train_split)
    num_test = len(dataset) - num_train

    train_dataset, test_dataset = random_split(
        dataset,
        [num_train, num_test],
        seed=config.seed
    )

    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    # Training history
    history = {'train_loss': [], 'train_mae': [], 'val_loss': [], 'val_mae': [], 'epoch': [], 'lr': []}
    best_val_mae = float('inf')
    best_epoch = 0

    logger.info("")
    logger.info("Starting training...")

    # Training loop
    for epoch in range(1, config.num_epochs + 1):
        lr = get_lr(epoch, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        train_errors = train_one_epoch(model, train_loader, optimizer, config.device, epoch, config)
        val_errors = validate(model, test_loader, config.device, config)

        history['epoch'].append(epoch)
        history['train_loss'].append(train_errors.get('loss', float('nan')))
        history['train_mae'].append(train_errors.get('hamiltonian_mae', float('nan')))
        history['val_loss'].append(val_errors.get('loss', float('nan')))
        history['val_mae'].append(val_errors.get('hamiltonian_mae', float('nan')))
        history['lr'].append(lr)

        val_mae = val_errors.get('hamiltonian_mae', float('nan'))
        is_best = not np.isnan(val_mae) and val_mae < best_val_mae

        if is_best:
            best_val_mae = val_mae
            best_epoch = epoch

        if epoch % config.log_interval == 0 or epoch == 1:
            train_loss = train_errors.get('loss', float('nan'))
            train_mae = train_errors.get('hamiltonian_mae', float('nan'))
            val_loss = val_errors.get('loss', float('nan'))
            val_mae_log = val_errors.get('hamiltonian_mae', float('nan'))
            logger.info(
                f"Epoch {epoch:5d} | "
                f"Train MAE: {train_mae:.6f} | "
                f"Val MAE: {val_mae_log:.6f} | "
                f"LR: {lr:.2e}"
            )
            if is_best:
                logger.info(f"  ✓ New best model saved at epoch {epoch}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': vars(config),
            'history': history,
            'best_val_mae': best_val_mae,
            'best_epoch': best_epoch,
        }

        if is_best:
            torch.save(checkpoint, log_dir / 'best_checkpoint.pth')
            logger.info(f"New best model at epoch {epoch} (MAE: {best_val_mae:.6f})")

        # Visualization and latest checkpoint every save_interval epochs.
        if epoch % config.save_interval == 0 or epoch == config.num_epochs:
            visualize_predictions(model, train_loader, test_loader, config.device, log_dir, epoch, logger)
            torch.save(checkpoint, log_dir / 'latest_checkpoint.pth')
            plot_training_curves(history, log_dir, logger, config.data_fraction)

    logger.info("")
    logger.info("Training completed!")
    logger.info(f"Best Val MAE: {best_val_mae:.6f} at epoch {best_epoch}")

    # Final training curves
    plot_training_curves(history, log_dir, logger, config.data_fraction)

    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': vars(config),
        'history': history,
        'best_val_mae': best_val_mae,
        'best_epoch': best_epoch,
    }, log_dir / 'final_model.pth')


if __name__ == '__main__':
    main()
