import torch


def soft_inlier_fun_gen(beta, tau, inverse=False):
    def f(d):
        if inverse:
            return torch.sigmoid(beta * d / tau - beta)
        else:
            return 1 - torch.sigmoid(beta * d / tau - beta)
    return f


def hard_inlier_fun_gen(beta, tau, inverse=False):
    def f(d):
        inlier = (d < tau).float()
        return 1-inlier if inverse else inlier
    return f


inlier_functions = {"hard": hard_inlier_fun_gen, "soft": soft_inlier_fun_gen}


def count_inliers(opt, residuals, log_weights):

    weights = torch.softmax(log_weights, dim=2)
    B, N, Mo = weights.size()
    _, K, S, M, N = residuals.size()
    weights_e = weights[..., :M].transpose(1, 2).view(B, 1, 1, M, N).expand(B, K, S, M, N)

    inlier_fun = inlier_functions[opt.inlier_function](opt.inlier_softness, opt.inlier_threshold)

    inlier_scores = inlier_fun(residuals)

    if opt.inlier_counting == "unweighted":
        inlier_scores_weighted = inlier_scores
    else:
        inlier_scores_weighted = inlier_scores * weights_e

    inlier_ratios = inlier_scores_weighted.sum(-1) * 1.0 / N

    return inlier_ratios, inlier_scores


def compute_cumulative_inliers(opt, scores):

    B, K, M, H, N = scores.size()

    combined_scores = torch.zeros_like(scores, device=scores.device)
    for mi in range(M):
        if mi == 0:
            combined_scores[:, :, mi] = scores[:, :, mi]
        else:
            combined_scores[:, :, mi] = torch.max(scores[:, :, 0:(mi+1)], dim=2)[0] * (opt.cumulative_loss ** mi)

    inlier_loss = -combined_scores.sum((2, 4)) * 1.0 / N

    return inlier_loss


def combine_hypotheses_inliers(inlier_scores):

    B, K, M, H, N = inlier_scores.size()

    combined_inliers = torch.max(inlier_scores, dim=2)[0]

    inlier_counts = combined_inliers.sum(-1) * 1.0 / N  # B, K, H

    return inlier_counts
