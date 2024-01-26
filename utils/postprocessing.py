import torch

from utils.residual_functions import compute_residuals
from utils.inlier_counting import inlier_functions


def assign_cluster_labels(opt, residuals, counts):
    if len(residuals.size()) == 5:
        mask = (counts < opt.mss).float()[..., None].transpose(2, 3)
        res = residuals.transpose(2, 3)
        B, K, M, H, N = residuals.size()
        batch_dims = [B, K, H]
    else:
        mask = (counts < opt.mss).float()[..., None]
        res = residuals
        B, K, M, N = residuals.size()
        batch_dims = [B, K]

    estm_labels = -1 * torch.ones(batch_dims + [N], device=residuals.device, dtype=torch.long)
    ones = torch.ones(batch_dims + [N], device=residuals.device, dtype=torch.long)
    min_dists = torch.ones(batch_dims + [N], device=residuals.device, dtype=torch.float32) * opt.inlier_threshold

    res = res + mask * 1000.0

    for mi in range(M):
        condition = torch.logical_and(res[..., mi, :] < opt.assignment_threshold, estm_labels == -1) | \
                    (res[..., mi, :] < min_dists)

        estm_labels = torch.where(condition, ones * mi, estm_labels)
        min_dists = torch.minimum(min_dists, res[..., mi, :])

    estm_labels += 1

    estm_clusters = torch.zeros(batch_dims + [M+1, N], dtype=torch.bool, device=residuals.device)
    for mi in range(M+1):
        estm_clusters[..., mi, :] = torch.where(estm_labels == mi, ones.bool(), estm_clusters[..., mi, :])

    return estm_labels, estm_clusters


def ranking_and_clustering(opt, inlier_scores, hypotheses, residuals):
    scores = torch.clone(inlier_scores.squeeze(2))

    all_best_models = []
    all_best_counts = []

    if len(scores.size()) == 4:
        B, K, M, N = scores.size()
        selected_scores = torch.zeros((B, K, 1, N), dtype=torch.float32, device=scores.device)
    elif len(scores.size()) == 5:
        B, K, M, H, N = scores.size()
        selected_scores = torch.zeros((B, K, 1, H, N), dtype=torch.float32, device=scores.device)
    else:
        assert False

    for mi in range(M):
        overlap = scores * selected_scores
        unique = scores * (1-selected_scores)

        ranked_counts_per_model = unique.sum(-1) - overlap.sum(-1)

        if len(all_best_models) > 0:
            bm = torch.concatenate(all_best_models, dim=2)
            ranked_counts_per_model.scatter_(2, bm, torch.ones_like(ranked_counts_per_model, device=ranked_counts_per_model.device) * (-1e8))

        best_counts, best_models = torch.max(ranked_counts_per_model, dim=2, keepdim=True)

        all_best_models += [best_models]
        all_best_counts += [best_counts]

        if len(scores.size()) == 4:
            best_scores = torch.gather(scores, 2, best_models.unsqueeze(-1).expand(B, K, 1, N))

        elif len(scores.size()) == 5:
            best_scores = torch.gather(scores, 2, best_models.unsqueeze(-1).expand(B, K, 1, H, N))
        else:
            assert False

        selected_scores = torch.maximum(best_scores, selected_scores)

    models = torch.concatenate(all_best_models, dim=2)
    ranked_counts_per_model = torch.concatenate(all_best_counts, dim=2)

    D = hypotheses.size(-1)
    ranked_hypotheses = torch.gather(hypotheses, 2, models.view(B, K, M, H, 1).expand(B, K, M, H, D))
    ranked_residuals = torch.gather(residuals, 2, models.view(B, K, M, H, 1).expand(B, K, M, H, N))
    ranked_scores = torch.gather(inlier_scores, 2, models.view(B, K, M, H, 1).expand(B, K, M, H, N))
    total_counts = selected_scores.sum(-1)

    if not opt.problem == "vp":
        labels, clusters = assign_cluster_labels(opt, ranked_residuals, ranked_counts_per_model)
    else:
        labels = None
        clusters = None

    return models, total_counts, ranked_hypotheses, ranked_scores, labels, clusters


def refinement_with_inliers(opt, X, inliers):
    B, K, M, H, N = inliers.size()

    if opt.problem == "vp":
        lines = X[:, :, 6:9]
        lines_repeated = lines.view(B, 1, 1, 1, N, 3).expand(B, K, M, H, N, 3)
        weights_repeated = inliers.view(B, K, M, H, N, 1).expand(B, K, M, H, N, 3)
        Mat = (lines_repeated * weights_repeated)
        MatSq = Mat.transpose(-1, -2) @ Mat
        _, _, V = torch.svd(MatSq)
        models = V[..., 2]

    else:
        assert False, "refinement for %s is not implemented yet" % opt.problem

    residuals = compute_residuals(opt, X, models)

    inlier_fun = inlier_functions[opt.inlier_function](opt.inlier_softness, opt.inlier_threshold)

    inlier_scores = inlier_fun(residuals)

    return models, residuals, inlier_scores