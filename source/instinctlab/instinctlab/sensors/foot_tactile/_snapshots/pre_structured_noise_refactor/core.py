from __future__ import annotations

from typing import Optional

import torch


def compute_foot_normal_w(
    quat_w: torch.Tensor,
    local_normal: torch.Tensor,
    quat_apply_func,
) -> torch.Tensor:
    """Map the local foot normal to world frame for all envs and feet."""

    expanded = local_normal.view(1, 1, 3).expand(quat_w.shape[0], quat_w.shape[1], 3)
    return quat_apply_func(quat_w, expanded)


def build_taxel_pos_w(
    taxel_xy_b: torch.Tensor,
    z_offset: float,
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    transform_points_func,
) -> torch.Tensor:
    """Transform local planar taxels into world frame."""

    num_bodies, max_taxels, _ = taxel_xy_b.shape
    taxel_xyz_b = torch.zeros((num_bodies, max_taxels, 3), device=taxel_xy_b.device, dtype=taxel_xy_b.dtype)
    taxel_xyz_b[..., :2] = taxel_xy_b
    taxel_xyz_b[..., 2] = z_offset

    num_envs = pos_w.shape[0]
    local_points = taxel_xyz_b.unsqueeze(0).expand(num_envs, -1, -1, -1).reshape(num_envs * num_bodies, max_taxels, 3)
    world_pos = pos_w.reshape(num_envs * num_bodies, 3)
    world_quat = quat_w.reshape(num_envs * num_bodies, 4)

    return transform_points_func(local_points, world_pos, world_quat).reshape(num_envs, num_bodies, max_taxels, 3)


def compute_bandexp_weights(
    support_dist: torch.Tensor,
    support_valid_mask: torch.Tensor,
    band: float,
    rho: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute band-limited exponential support weights."""

    device = support_dist.device
    dtype = support_dist.dtype

    dist = torch.nan_to_num(support_dist, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
    valid = support_valid_mask & torch.isfinite(dist)

    b = max(float(band), eps)
    delta = 0.01 * b
    ln_rho = torch.log(torch.tensor(float(rho), device=device, dtype=dtype)).clamp_min(1e-6)

    in_band = valid & (dist <= b)
    has_contact = in_band.any(dim=-1, keepdim=True)

    d_for_max = torch.where(in_band, dist, torch.tensor(float("-inf"), device=device, dtype=dtype))
    d_for_min = torch.where(in_band, dist, torch.tensor(float("inf"), device=device, dtype=dtype))
    D = d_for_max.max(dim=-1, keepdim=True).values
    d0 = d_for_min.min(dim=-1, keepdim=True).values

    R = (D - d0).clamp_min(0.0)
    tau_default = torch.tensor(b, device=device, dtype=dtype) / ln_rho
    tau = torch.where(R > eps, R / ln_rho, tau_default.expand_as(R))

    weights = torch.exp(-(dist - d0) / (tau + eps))
    taper = torch.clamp((b + delta - dist) / max(delta, 1e-12), 0.0, 1.0)

    weights = weights * taper * valid.to(dtype=dtype)
    weights = torch.where(has_contact, weights, torch.zeros_like(weights))
    weights = torch.clamp(weights, min=0.0)
    return torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)


def apply_alignment_gating(
    weight_bandexp: torch.Tensor,
    support_alignment: Optional[torch.Tensor],
    support_valid_mask: torch.Tensor,
    a0: float = 0.3,
    q: float = 1.5,
    align_mix: float = 0.4,
) -> torch.Tensor:
    """Apply smooth alignment gating on top of band-exp weights."""

    weights = torch.nan_to_num(weight_bandexp, nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.clamp(weights, min=0.0)
    valid = support_valid_mask.to(dtype=weights.dtype)

    if support_alignment is None:
        return weights * valid

    a0 = float(min(max(a0, -0.999), 0.999999))
    q = float(max(q, 0.0))
    align_mix = float(min(max(align_mix, 0.0), 1.0))
    denom = max(1.0 - a0, 1e-6)

    align = torch.nan_to_num(support_alignment, nan=a0, posinf=1.0, neginf=-1.0)
    g_align = torch.clamp((align - a0) / denom, 0.0, 1.0).pow(q)
    gated = weights * ((1.0 - align_mix) + align_mix * g_align)
    gated = torch.clamp(gated, min=0.0) * valid
    return torch.nan_to_num(gated, nan=0.0, posinf=0.0, neginf=0.0)


def normalize_taxel_weights(
    weights: torch.Tensor,
    valid_taxel_mask: torch.Tensor,
    eps: float = 1e-8,
    fallback_to_uniform: bool = False,
) -> torch.Tensor:
    """Normalize per-foot taxel weights to a sum of 1."""

    safe = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    safe = torch.clamp(safe, min=0.0)
    valid = valid_taxel_mask.unsqueeze(0).to(dtype=safe.dtype)
    safe = safe * valid

    denom = safe.sum(dim=-1, keepdim=True)
    normalized = torch.where(denom > eps, safe / denom, torch.zeros_like(safe))

    if fallback_to_uniform:
        valid_count = valid.sum(dim=-1, keepdim=True)
        uniform = torch.where(valid_count > 0.0, valid / valid_count.clamp_min(1.0), torch.zeros_like(valid))
        normalized = torch.where(denom > eps, normalized, uniform)

    normalized = normalized * valid
    normalized = torch.clamp(normalized, min=0.0)
    return torch.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)


def distribute_total_force_to_taxels(
    total_normal_force: torch.Tensor,
    weights: torch.Tensor,
    valid_taxel_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Distribute per-foot total normal force while preserving per-foot sums."""

    total = torch.nan_to_num(total_normal_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    normalized = normalize_taxel_weights(weights, valid_taxel_mask, eps=eps, fallback_to_uniform=True)

    has_force = total.unsqueeze(-1) > eps
    normalized = torch.where(has_force, normalized, torch.zeros_like(normalized))

    valid = valid_taxel_mask.unsqueeze(0).to(dtype=normalized.dtype)
    force = total.unsqueeze(-1) * normalized
    force = torch.clamp(force, min=0.0) * valid
    force = _match_force_totals(force, target_total=total.unsqueeze(-1), valid=valid, eps=eps)
    return torch.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)


def build_knn_diffusion_matrix(
    taxel_xy_b: torch.Tensor,
    valid_taxel_mask: torch.Tensor,
    knn: int,
) -> torch.Tensor:
    """Build a per-foot kNN diffusion matrix P with shape (B, T, T)."""

    num_bodies, max_taxels, _ = taxel_xy_b.shape
    matrix = torch.zeros((num_bodies, max_taxels, max_taxels), device=taxel_xy_b.device, dtype=taxel_xy_b.dtype)

    k = max(int(knn), 0)
    for body_id in range(num_bodies):
        valid_idx = torch.nonzero(valid_taxel_mask[body_id], as_tuple=False).squeeze(-1)
        num_valid = int(valid_idx.numel())
        if num_valid == 0:
            continue
        if num_valid == 1 or k == 0:
            matrix[body_id, valid_idx, valid_idx] = 1.0
            continue

        points = taxel_xy_b[body_id, valid_idx]
        pairwise_dist = torch.cdist(points, points, p=2.0)
        pairwise_dist.fill_diagonal_(float("inf"))

        k_eff = min(k, num_valid - 1)
        knn_local = torch.topk(pairwise_dist, k=k_eff, dim=-1, largest=False).indices
        row_idx = valid_idx.unsqueeze(-1).expand(-1, k_eff)
        col_idx = valid_idx[knn_local]
        matrix[body_id, row_idx, col_idx] = 1.0 / float(k_eff)

    matrix = torch.clamp(matrix, min=0.0)
    return torch.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def diffuse_taxel_force_knn(
    taxel_force_clean: torch.Tensor,
    diffusion_matrix: torch.Tensor,
    valid_taxel_mask: torch.Tensor,
    alpha: float,
    diffusion_iters: int,
    preserve_total_force: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Diffuse taxel force using F_new = (1-alpha)*F + alpha*(P@F)."""

    force = torch.nan_to_num(taxel_force_clean, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    valid = valid_taxel_mask.unsqueeze(0).to(dtype=force.dtype)
    force = force * valid

    alpha = float(min(max(alpha, 0.0), 1.0))
    num_iters = max(int(diffusion_iters), 0)
    if num_iters == 0 or alpha <= 0.0:
        return force

    P = torch.nan_to_num(diffusion_matrix, nan=0.0, posinf=0.0, neginf=0.0).to(dtype=force.dtype)
    P = torch.clamp(P, min=0.0)

    target_total = force.sum(dim=-1, keepdim=True, dtype=torch.float64).to(dtype=force.dtype)
    for _ in range(num_iters):
        neighbor_force = torch.einsum("bij,ebj->ebi", P, force)
        neighbor_force = torch.nan_to_num(neighbor_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)

        force = (1.0 - alpha) * force + alpha * neighbor_force
        force = torch.clamp(force, min=0.0) * valid

        if preserve_total_force:
            force = _match_force_totals(force, target_total=target_total, valid=valid, eps=eps)

    force = torch.where(target_total > eps, force, torch.zeros_like(force))
    force = torch.clamp(force, min=0.0) * valid
    return torch.nan_to_num(force, nan=0.0, posinf=0.0, neginf=0.0)


def _match_force_totals(
    force: torch.Tensor,
    target_total: torch.Tensor,
    valid: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Rescale per-foot force and correct tiny floating error residuals."""

    current_total = force.sum(dim=-1, keepdim=True, dtype=torch.float64).to(dtype=force.dtype)
    scale = torch.where(current_total > eps, target_total / current_total, torch.zeros_like(current_total))
    scaled = force * scale

    valid_count = valid.sum(dim=-1, keepdim=True)
    has_valid = valid_count > 0.0
    residual = target_total - scaled.sum(dim=-1, keepdim=True, dtype=torch.float64).to(dtype=scaled.dtype)
    residual = torch.where(has_valid, residual, torch.zeros_like(residual))

    if torch.any(has_valid):
        masked = torch.where(valid > 0.0, scaled, torch.full_like(scaled, -1.0))
        idx = masked.argmax(dim=-1, keepdim=True)
        scaled = scaled.scatter_add(-1, idx, residual)

    scaled = torch.where(target_total > eps, scaled, torch.zeros_like(scaled))
    scaled = torch.clamp(scaled, min=0.0) * valid
    return torch.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)


def compute_contact_area_ratio(
    taxel_force: torch.Tensor,
    valid_taxel_mask: torch.Tensor,
    active_threshold: float,
) -> torch.Tensor:
    """Fraction of valid taxels whose force exceeds the active threshold."""

    safe_force = torch.nan_to_num(taxel_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    active = (safe_force > active_threshold) & valid_taxel_mask.unsqueeze(0)
    denom = valid_taxel_mask.sum(dim=-1).unsqueeze(0).float().clamp_min(1.0)
    ratio = active.float().sum(dim=-1) / denom
    return torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)


def compute_edge_force_ratio(
    taxel_force: torch.Tensor,
    edge_taxel_mask: torch.Tensor,
    valid_taxel_mask: torch.Tensor,
) -> torch.Tensor:
    """Ratio of force carried by edge taxels."""

    safe_force = torch.nan_to_num(taxel_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    masked = safe_force * valid_taxel_mask.unsqueeze(0).float()
    edge = safe_force * edge_taxel_mask.unsqueeze(0).float()
    ratio = edge.sum(dim=-1) / (masked.sum(dim=-1) + 1e-6)
    return torch.nan_to_num(ratio, nan=0.0, posinf=0.0, neginf=0.0)


def compute_basic_force_stats(
    taxel_force: torch.Tensor,
    valid_taxel_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-foot peak and mean taxel force."""

    safe_force = torch.nan_to_num(taxel_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    masked = safe_force * valid_taxel_mask.unsqueeze(0).float()
    denom = valid_taxel_mask.sum(dim=-1).unsqueeze(0).float().clamp_min(1.0)
    mean_force = masked.sum(dim=-1) / denom
    peak_force = masked.max(dim=-1).values
    return (
        torch.nan_to_num(peak_force, nan=0.0, posinf=0.0, neginf=0.0),
        torch.nan_to_num(mean_force, nan=0.0, posinf=0.0, neginf=0.0),
    )


def compute_cop_b(
    taxel_force: torch.Tensor,
    taxel_xy_b: torch.Tensor,
    valid_taxel_mask: torch.Tensor,
) -> torch.Tensor:
    """Center of pressure in local foot plane."""

    safe_force = torch.nan_to_num(taxel_force, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    masked_force = safe_force * valid_taxel_mask.unsqueeze(0).float()
    weighted_xy = masked_force.unsqueeze(-1) * taxel_xy_b.unsqueeze(0)
    cop = weighted_xy.sum(dim=-2) / (masked_force.sum(dim=-1, keepdim=True) + 1e-6)
    return torch.nan_to_num(cop, nan=0.0, posinf=0.0, neginf=0.0)


def point_near_polygon_edge_mask(
    points_xy: torch.Tensor,
    polygon_xy: torch.Tensor,
    edge_margin: float,
) -> torch.Tensor:
    """Return True for points that lie within edge_margin of the polygon boundary."""

    if polygon_xy.shape[0] < 2:
        raise ValueError("polygon_xy must contain at least two points.")

    if not torch.allclose(polygon_xy[0], polygon_xy[-1]):
        polygon_xy = torch.cat([polygon_xy, polygon_xy[:1]], dim=0)

    seg_start = polygon_xy[:-1]
    seg_end = polygon_xy[1:]

    points = points_xy.unsqueeze(1)
    start = seg_start.unsqueeze(0)
    end = seg_end.unsqueeze(0)

    seg = end - start
    rel = points - start
    denom = (seg * seg).sum(dim=-1, keepdim=True).clamp_min(1e-8)
    t = ((rel * seg).sum(dim=-1, keepdim=True) / denom).clamp(0.0, 1.0)
    closest = start + t * seg
    dist = torch.norm(points - closest, dim=-1)

    return dist.min(dim=1).values <= edge_margin
