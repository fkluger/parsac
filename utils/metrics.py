import itertools
import math

import numpy as np
import scipy.optimize
import torch

from utils import residual_functions


def hungarian_loss(cost, num_models=None, max_error=None):
    B, K, H, M1, M2 = cost.size()
    losses = np.zeros((B, K, H), dtype=np.float32)

    cost_np = cost.cpu().detach().numpy()
    for bi in range(B):
        for ki in range(K):
            for hi in range(H):
                if num_models is not None:
                    num_true = num_models[bi]
                    cost_mat = cost_np[bi, ki, hi, :num_true, :num_true]
                else:
                    cost_mat = cost_np[bi, ki, hi]
                row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost_mat)
                errors = cost_mat[row_ind, col_ind]
                if max_error is None:
                    loss = errors.sum()
                else:
                    loss = np.clip(errors, a_max=max_error, a_min=None).sum()

                losses[bi, ki, hi] = loss
    return torch.from_numpy(losses).to(cost.device)


def hungarian_loss_torch(cost, num_models=None, max_error=None, max_gpu_alloc=6):
    B, K, H, M1, M2 = cost.size()
    if M1 > M2:
        cost = cost.transpose(-1, -2)
        B, K, H, M1, M2 = cost.size()

    perm = itertools.permutations(range(M2), M1)
    perm_list = list(perm)
    P = len(perm_list)

    if B * K * H * P * M1 * 4 > max_gpu_alloc * (1024 ** 3):
        return hungarian_loss(cost, num_models, max_error)

    perm_list = np.array(perm_list)

    range1 = torch.arange(0, M1).view(1, M1).expand(B, M1).to(cost.device)
    range2 = torch.arange(0, M2).view(1, M2).expand(B, M2).to(cost.device)

    if num_models is not None:
        mask1 = range1 < num_models.view(B, 1)
        mask2 = range2 < num_models.view(B, 1)
        mask = torch.logical_and(mask1.view(B, M1, 1), mask2.view(B, 1, M2)).view(B, 1, 1, M1, M2).float()
        cost = cost * mask
        mask = torch.logical_xor(mask1.view(B, M1, 1), mask2.view(B, 1, M2)).view(B, 1, 1, M1, M2).float()
        cost = cost + mask * 10000.0

    permutations = torch.from_numpy(perm_list).to(cost.device)
    permutations = permutations.view(1, 1, 1, P, M1, 1).expand(B, K, H, P, M1, 1)

    cost_e = cost.view(B, K, H, 1, M1, M2).expand(B, K, H, P, M1, M2)

    selected_cost = torch.gather(cost_e, -1, permutations).squeeze(-1)

    cost_sums = selected_cost.sum(-1)
    min_idx = torch.argmin(cost_sums, dim=3, keepdim=True)

    min_costs = torch.gather(selected_cost, 3, min_idx.view(B, K, H, 1, 1).expand(B, K, H, 1, M1))

    if max_error is not None:
        losses = torch.clip(min_costs, max=max_error).sum(-1).squeeze(-1)
    else:
        losses = min_costs.sum(-1).squeeze(-1)

    return losses, min_costs


def vp_loss(true_vps, estm_vps, inv_intrinsics, max_error=10.):
    B, Mt, D = true_vps.size()
    B, K, M, H, D = estm_vps.size()

    inv_intrinsics = inv_intrinsics.to(true_vps.device)

    true_vds = (inv_intrinsics @ true_vps.unsqueeze(-1)).squeeze(-1)
    estm_vds = (inv_intrinsics @ estm_vps.unsqueeze(-1)).squeeze(-1)

    true_vds = true_vds / torch.clip(torch.norm(true_vds, dim=-1, keepdim=True), min=1e-8)
    estm_vds = estm_vds / torch.clip(torch.norm(estm_vds, dim=-1, keepdim=True), min=1e-8)

    estm_vds = estm_vds.transpose(2, 3).view(B, K, H, 1, M, D)
    true_vds = true_vds.view(B, 1, 1, Mt, 1, D)
    cosines = (true_vds * estm_vds).sum(-1)
    cosines = torch.clip(cosines, min=-1.0 + 1e-8, max=1.0 - 1e-8)

    cost_matrix = torch.arccos(torch.abs(cosines)) * 180. / math.pi

    true_vps_norms = torch.norm(true_vps, dim=-1)

    losses, min_costs = hungarian_loss_torch(cost_matrix, (true_vps_norms > 1e-8).sum(-1), max_error)

    return losses, min_costs


def classification_loss(opt, true_labels, clusters):
    with torch.no_grad():

        inliers_only = opt.ablation_outlier_ratio >= 0
        if inliers_only:
            num_true_inliers = (true_labels > 0).sum(-1)[0]
            true_labels = true_labels[..., :num_true_inliers]
            clusters = clusters[..., :num_true_inliers]

        B, K, H, Mo, N = clusters.size()
        M = opt.instances

        num_true_labels = torch.max(true_labels).item() + 1

        true_clusters = torch.zeros((B, num_true_labels, N), dtype=torch.bool, device=true_labels.device)

        for li in range(num_true_labels):
            true_clusters[:, li] = torch.where(true_labels == li,
                                               torch.ones((B, N), dtype=torch.bool, device=true_labels.device),
                                               true_clusters[:, li])

        true_clusters = true_clusters.view(B, 1, 1, num_true_labels, N)

        correct_assignments = torch.logical_and(clusters[..., None, :], true_clusters[..., None, :, :])

        num_correct_per_class = correct_assignments.int().sum(-1)
        if num_true_labels > 5 or M > 9:
            num_correct_total = -hungarian_loss(-num_correct_per_class)
        else:
            num_correct_total, _ = hungarian_loss_torch(-num_correct_per_class)
            num_correct_total = -num_correct_total

        losses = (true_clusters.sum((-1, -2)) - num_correct_total) / true_clusters.sum((-1, -2))

        return losses


def geometric_errors(opt, X, image_size, ground_truth, models):
    x = torch.clone(X[..., :4])

    scale = torch.max(image_size, dim=-1)[0]

    x[..., 0:2] *= scale[:, 0][:, None, None] / 2.0
    x[..., 2:4] *= scale[:, 1][:, None, None] / 2.0

    x[..., 0] += image_size[..., 0, 1][:, None] / 2.0
    x[..., 1] += image_size[..., 0, 0][:, None] / 2.0
    x[..., 2] += image_size[..., 1, 1][:, None] / 2.0
    x[..., 3] += image_size[..., 1, 0][:, None] / 2.0

    B, N, C = x.size()
    _, K, M, D = models.size()

    xe = x.view(B, 1, 1, N, C).expand(B, K, M, N, C)

    if opt.problem == "fundamental":
        T1 = torch.zeros((B, 3, 3), device=X.device)
        T2 = torch.zeros((B, 3, 3), device=X.device)

        T1[:, 0, 0] = 2.0 / scale[:, 0]
        T1[:, 1, 1] = 2.0 / scale[:, 0]
        T1[:, 2, 2] = 1

        T2[:, 0, 0] = 2.0 / scale[:, 1]
        T2[:, 1, 1] = 2.0 / scale[:, 1]
        T2[:, 2, 2] = 1

        T1[:, 0, 2] = -image_size[..., 0, 1] / scale[:, 0]
        T1[:, 1, 2] = -image_size[..., 0, 0] / scale[:, 0]
        T2[:, 0, 2] = -image_size[..., 1, 1] / scale[:, 1]
        T2[:, 1, 2] = -image_size[..., 1, 0] / scale[:, 1]

        T1 = T1[:, None, None, ...]
        T2 = T2[:, None, None, ...].transpose(-1, -2)

        F = torch.clone(models.view(B, K, M, 3, 3))
        F = T2 @ F @ T1

        m = F.view(B, K, M, 9)

    elif opt.problem == "homography":
        T1 = torch.zeros((B, 3, 3), device=X.device)
        T2 = torch.zeros((B, 3, 3), device=X.device)

        T1[:, 0, 0] = 2.0 / scale[:, 0]
        T1[:, 1, 1] = 2.0 / scale[:, 0]
        T1[:, 2, 2] = 1

        T2[:, 0, 0] = scale[:, 1] / 2.0
        T2[:, 1, 1] = scale[:, 1] / 2.0
        T2[:, 2, 2] = 1

        T1[:, 0, 2] = -image_size[..., 0, 1] / scale[:, 0]
        T1[:, 1, 2] = -image_size[..., 0, 0] / scale[:, 0]
        T2[:, 0, 2] = image_size[..., 1, 1] / 2.0
        T2[:, 1, 2] = image_size[..., 1, 0] / 2.0

        T1 = T1[:, None, None, ...]
        T2 = T2[:, None, None, ...]

        H = torch.clone(models.view(B, K, M, 3, 3))
        H = T2 @ H @ T1

        m = H.view(B, K, M, 9)

    else:
        m = models

    r = residual_functions.mapping[opt.problem](xe, m)

    if opt.problem == "homography":
        r = torch.sqrt(r)

    r = torch.clamp(r, min=None, max=torch.max(image_size))

    r = r.detach().cpu().numpy()

    r_avg = np.zeros((B, K))

    gt = ground_truth.cpu().numpy()

    for bi in range(B):
        for ki in range(K):
            all_selected_ge = []
            num_classes = gt[bi].max()
            for ci in range(num_classes):
                ge_sel = r[bi, ki, :, :][:num_classes, np.where(gt[bi] == ci + 1)[0]]

                selected_ge = np.min(ge_sel, axis=0)

                all_selected_ge += selected_ge.flatten().tolist()

            r_avg[bi, ki] = np.mean(all_selected_ge)

    return r_avg
