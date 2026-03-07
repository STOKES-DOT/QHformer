"""
数据处理工具 - 为DeepMolH-E3准备分子数据

提供数据加载、预处理、边构建等功能
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MolecularData:
    """分子数据结构"""
    atomic_numbers: torch.Tensor      # (num_atoms,)
    positions: torch.Tensor            # (num_atoms, 3)
    edge_indices: torch.Tensor         # (2, num_edges)
    edge_distances: torch.Tensor       # (num_edges,)
    hamiltonian: torch.Tensor          # (num_atoms, num_orbitals, num_orbitals)
    num_atoms: int
    num_orbitals: int


def build_edges(
    positions: torch.Tensor,
    cutoff: float = 5.0,
    self_interaction: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    构建分子图的边（基于距离截断）

    Args:
        positions: (num_atoms, 3) 原子坐标
        cutoff: 距离截断值（Å）
        self_interaction: 是否包含自相互作用

    Returns:
        edge_indices: (2, num_edges) 边索引
        edge_distances: (num_edges,) 边距离
    """
    num_atoms = positions.shape[0]

    # 计算距离矩阵
    diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (num_atoms, num_atoms, 3)
    distances = torch.norm(diff, dim=-1)  # (num_atoms, num_atoms)

    # 找到满足截断条件的边
    mask = (distances <= cutoff) & (distances > 0)
    if self_interaction:
        mask = mask | (torch.eye(num_atoms, device=positions.device).bool())

    # 获取边索引
    src, dst = torch.where(mask)
    edge_indices = torch.stack([src, dst])
    edge_distances = distances[src, dst]

    return edge_indices, edge_distances


def compute_hamiltonian_pyscf(
    atomic_numbers: torch.Tensor,
    positions: torch.Tensor,
    basis: str = "sto-3g",
) -> torch.Tensor:
    """
    使用PySCF计算哈密顿矩阵

    Args:
        atomic_numbers: (num_atoms,) 原子序数
        positions: (num_atoms, 3) 原子坐标（单位：Å）
        basis: 基组名称

    Returns:
        hamiltonian: (num_orbitals, num_orbitals) 哈密顿矩阵
    """
    try:
        from pyscf import gto, scf
    except ImportError:
        raise ImportError("PySCF is required for Hamiltonian calculation. Install with: pip install pyscf")

    # 转换为原子单位（Bohr）
    positions_bohr = positions / 0.52917721067  # Å to Bohr

    # 创建分子
    mol = gto.Mole()
    mol.atom = []
    for z, pos in zip(atomic_numbers.tolist(), positions_bohr.tolist()):
        mol.atom.append((z, pos))
    mol.basis = basis
    mol.build()

    # 计算哈密顿矩阵（使用HF方法）
    mf = scf.RHF(mol)
    mf.kernel()

    # 获取哈密顿矩阵（Core Hamiltonian = H_core = T + V）
    # 或者直接使用Fock矩阵
    hamiltonian = torch.from_numpy(mf.get_fock()).float()

    return hamiltonian


def create_random_molecule(
    num_atoms: int = 10,
    atom_types: List[int] = None,
    box_size: float = 10.0,
) -> Dict[str, torch.Tensor]:
    """
    创建随机分子用于测试

    Args:
        num_atoms: 原子数量
        atom_types: 原子类型列表，默认使用C, H, O, N
        box_size: 盒子大小（Å）

    Returns:
        包含分子信息的字典
    """
    if atom_types is None:
        atom_types = [1, 6, 7, 8]  # H, C, N, O

    # 随机生成原子类型和位置
    atomic_numbers = torch.tensor([
        np.random.choice(atom_types) for _ in range(num_atoms)
    ], dtype=torch.long)

    positions = torch.rand(num_atoms, 3) * box_size

    # 构建边
    edge_indices, edge_distances = build_edges(positions, cutoff=5.0)

    # 对于测试，创建随机哈密顿矩阵
    num_orbitals = num_atoms * 2  # 简化假设
    hamiltonian = torch.randn(num_atoms, num_orbitals, num_orbitals)
    # 使其对称
    hamiltonian = (hamiltonian + hamiltonian.transpose(-2, -1)) / 2

    return {
        'atomic_numbers': atomic_numbers,
        'positions': positions,
        'edge_indices': edge_indices,
        'edge_distances': edge_distances,
        'hamiltonian': hamiltonian,
    }


class MolecularDataset(torch.utils.data.Dataset):
    """
    分子数据集类
    """

    def __init__(
        self,
        data_list: List[Dict[str, torch.Tensor]],
        cutoff: float = 5.0,
    ):
        self.data_list = data_list
        self.cutoff = cutoff

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.data_list[idx]

        # 确保边是正确构建的
        if 'edge_indices' not in data:
            edge_indices, edge_distances = build_edges(
                data['positions'], self.cutoff
            )
            data['edge_indices'] = edge_indices
            data['edge_distances'] = edge_distances

        return data


def collate_molecular_data(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """
    将一批分子数据整理为一个batch

    使用padding处理不同大小的分子

    Args:
        batch: 分子数据列表

    Returns:
        整理后的batch
    """
    # 找到最大原子数
    max_atoms = max(item['atomic_numbers'].shape[0] for item in batch)
    max_edges = max(item['edge_indices'].shape[1] for item in batch)
    max_orbitals = max(item['hamiltonian'].shape[-1] for item in batch)

    batch_size = len(batch)

    # 初始化padding后的张量
    atomic_numbers = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    positions = torch.zeros(batch_size, max_atoms, 3)
    edge_indices = torch.zeros(batch_size, 2, max_edges, dtype=torch.long)
    edge_distances = torch.zeros(batch_size, max_edges)
    hamiltonians = torch.zeros(batch_size, max_atoms, max_orbitals, max_orbitals)
    masks = torch.zeros(batch_size, max_atoms, dtype=torch.bool)  # 有效原子mask

    for i, item in enumerate(batch):
        num_atoms = item['atomic_numbers'].shape[0]
        num_edges = item['edge_indices'].shape[1]
        num_orb = item['hamiltonian'].shape[-1]

        # 复制数据
        atomic_numbers[i, :num_atoms] = item['atomic_numbers']
        positions[i, :num_atoms] = item['positions']
        edge_indices[i, :, :num_edges] = item['edge_indices']
        edge_distances[i, :num_edges] = item['edge_distances']
        hamiltonians[i, :num_atoms, :num_orb, :num_orb] = item['hamiltonian']
        masks[i, :num_atoms] = True

    return {
        'atomic_numbers': atomic_numbers,
        'positions': positions,
        'edge_indices': edge_indices,
        'edge_distances': edge_distances,
        'hamiltonian': hamiltonians,
        'mask': masks,
    }


def create_water_oligomer_dataset(
    num_molecules: int = 100,
    max_water_units: int = 5,
    seed: int = 42,
) -> List[Dict[str, torch.Tensor]]:
    """
    创建水分子寡聚物数据集（用于测试）

    Args:
        num_molecules: 分子数量
        max_water_units: 最大水分子单元数
        seed: 随机种子

    Returns:
        分子数据列表
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_list = []

    for _ in range(num_molecules):
        # 随机决定水分子数量
        num_water = np.random.randint(2, max_water_units + 1)
        num_atoms = num_water * 3  # 每个水分子有3个原子（H2O）

        # 创建水分子
        atomic_numbers = []
        positions = []

        for i in range(num_water):
            # H2O: O-H-H
            atomic_numbers.extend([8, 1, 1])  # O, H, H

            # 随机取向（简化）
            center = np.random.rand(3) * 10

            # 氧原子位置
            o_pos = center
            # 氢原子位置（相对于氧，简化为固定键长和角度）
            h_dist = 0.96  # O-H键长（Å）
            h_angle = 104.5 * np.pi / 180  # H-O-H角度

            h1_pos = o_pos + np.array([
                h_dist * np.sin(h_angle/2),
                h_dist * np.cos(h_angle/2),
                0
            ])
            h2_pos = o_pos + np.array([
                -h_dist * np.sin(h_angle/2),
                h_dist * np.cos(h_angle/2),
                0
            ])

            positions.extend([o_pos, h1_pos, h2_pos])

        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)
        positions = torch.tensor(positions, dtype=torch.float32)

        # 构建边
        edge_indices, edge_distances = build_edges(positions, cutoff=5.0)

        # 创建随机哈密顿矩阵（实际应用中应使用PySCF计算）
        num_orbitals = num_atoms * 2
        hamiltonian = torch.randn(num_atoms, num_orbitals, num_orbitals)
        hamiltonian = (hamiltonian + hamiltonian.transpose(-2, -1)) / 2

        data_list.append({
            'atomic_numbers': atomic_numbers,
            'positions': positions,
            'edge_indices': edge_indices,
            'edge_distances': edge_distances,
            'hamiltonian': hamiltonian,
        })

    return data_list


# 使用示例
if __name__ == "__main__":
    print("创建测试数据集...")

    # 创建水分子寡聚物数据集
    dataset = create_water_oligomer_dataset(num_molecules=10)
    print(f"创建了 {len(dataset)} 个分子")

    # 创建DataLoader
    from torch.utils.data import DataLoader

    dataset_obj = MolecularDataset(dataset)
    dataloader = DataLoader(
        dataset_obj,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_molecular_data,
    )

    # 测试加载
    for batch in dataloader:
        print(f"\nBatch 信息:")
        print(f"  atomic_numbers: {batch['atomic_numbers'].shape}")
        print(f"  positions: {batch['positions'].shape}")
        print(f"  edge_indices: {batch['edge_indices'].shape}")
        print(f"  edge_distances: {batch['edge_distances'].shape}")
        print(f"  hamiltonian: {batch['hamiltonian'].shape}")
        print(f"  mask: {batch['mask'].shape}")
        break

    print("\n数据加载测试完成！")
