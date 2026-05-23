"""Torch-only replacements for the small PyG extension surface QHformer uses."""

import torch


def _expand_index(index, src, dim):
    if index.dim() == 1:
        index_shape = [1] * src.dim()
        index_shape[dim] = index.shape[0]
        return index.reshape(index_shape).expand_as(src)
    return index.expand_as(src)


def _empty_scatter_out(src, dim, dim_size, fill_value=0):
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    return torch.full(out_shape, fill_value, dtype=src.dtype, device=src.device)


def scatter(src, index, dim=0, dim_size=None, reduce="sum", out=None):
    """Subset-compatible torch_scatter.scatter replacement.

    QHformer only needs differentiable scatter reductions for attention
    aggregation and normalization. This implementation intentionally keeps that
    surface small and depends only on native PyTorch operations.
    """
    if dim < 0:
        dim = src.dim() + dim
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0

    index_expanded = _expand_index(index.to(torch.long), src, dim)
    reduce = "sum" if reduce == "add" else reduce

    if out is None:
        out = _empty_scatter_out(src, dim, dim_size)
    else:
        out.zero_()

    if reduce == "sum":
        return out.scatter_add_(dim, index_expanded, src)

    if reduce == "mean":
        counts = torch.zeros_like(out)
        out.scatter_add_(dim, index_expanded, src)
        counts.scatter_add_(dim, index_expanded, torch.ones_like(src))
        return out / counts.clamp(min=1)

    if reduce in {"max", "min"}:
        if reduce == "max":
            fill_value = -torch.inf if src.is_floating_point() else torch.iinfo(src.dtype).min
            torch_reduce = "amax"
        else:
            fill_value = torch.inf if src.is_floating_point() else torch.iinfo(src.dtype).max
            torch_reduce = "amin"
        reduced = _empty_scatter_out(src, dim, dim_size, fill_value)
        reduced.scatter_reduce_(dim, index_expanded, src, reduce=torch_reduce, include_self=True)
        counts = torch.zeros_like(reduced)
        counts.scatter_add_(dim, index_expanded, torch.ones_like(src))
        return torch.where(counts > 0, reduced, torch.zeros_like(reduced))

    raise ValueError(f"Unsupported scatter reduce={reduce!r}")


def scatter_add(src, index, dim=0, dim_size=None, out=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum", out=out)


def scatter_mean(src, index, dim=0, dim_size=None, out=None):
    return scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean", out=out)


def _rank_by_first_row(edge_index):
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)
    dst = edge_index[0]
    positions = torch.arange(dst.numel(), device=edge_index.device)
    is_start = torch.ones_like(dst, dtype=torch.bool)
    is_start[1:] = dst[1:] != dst[:-1]
    starts = torch.where(is_start, positions, torch.zeros_like(positions))
    starts = torch.cummax(starts, dim=0).values
    return positions - starts


def radius_graph(pos, r, batch=None, loop=False, max_num_neighbors=None, flow="source_to_target"):
    """Torch-only directed radius graph.

    The returned convention is ``edge_index = [dst, src]``, matching the
    existing QHformer edge-vector code. Edges are generated independently per
    graph in ``batch`` and never connect different molecules.
    """
    if flow not in {"source_to_target", "target_to_source"}:
        raise ValueError(f"Unsupported flow={flow!r}")
    num_nodes = pos.shape[0]
    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=pos.device)
    if num_nodes == 0:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)

    graph_ids = torch.unique(batch, sorted=True)
    edge_indices = []
    radius = float(r)
    for graph_idx in graph_ids.tolist():
        node_idx = torch.nonzero(batch == graph_idx, as_tuple=False).squeeze(-1)
        node_pos = pos.index_select(0, node_idx)
        if node_pos.numel() == 0:
            continue

        dist = torch.cdist(node_pos, node_pos)
        adj = dist <= radius
        if not loop:
            adj.fill_diagonal_(False)

        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        if edge_index.numel() == 0:
            continue
        if max_num_neighbors is not None and max_num_neighbors > 0:
            keep = _rank_by_first_row(edge_index) < int(max_num_neighbors)
            edge_index = edge_index[:, keep]
        if flow == "target_to_source":
            edge_index = edge_index.flip(0)
        edge_index = node_idx.index_select(0, edge_index.reshape(-1)).reshape(2, -1)
        edge_indices.append(edge_index)

    if not edge_indices:
        return torch.empty(2, 0, dtype=torch.long, device=pos.device)
    return torch.cat(edge_indices, dim=1)
