import torch


def expected_self_losses(opt, inlier_counts, cumulative_losses):
    if opt.cumulative_loss > -1:
        return cumulative_losses
    else:
        return -inlier_counts


def log_probabilities(log_sample_weights, choices, log_p_M_S):
    # implements Eq. 10
    B, K, S, M, mss = choices.size()
    B, N, Mo = log_sample_weights.size()

    log_p_x_j = log_sample_weights.transpose(1, 2).view(B, 1, 1, Mo, N).expand(B, K, S, Mo, N)

    log_p_h = torch.gather(log_p_x_j[..., :M, :], -1, choices)  # B, K, S, M, mss
    log_p_h = log_p_h.sum(-1)  # B, K, S, M

    log_p_S = log_p_h.sum(2)  # B, K, M

    log_p_M = log_p_M_S + log_p_S.sum(2).view(B, K, 1)

    return log_p_M


def backward_pass(opt, losses, log_p_M, optimizer):
    if opt.hypsamples > 0:
        baselines = losses.mean((-1, -2), keepdim=True)
        outputs = [log_p_M]
        gradients = [(losses - baselines).detach()]

    else:
        baselines = losses.mean(-1, keepdim=True)

        outputs = [log_p_M, losses]
        gradients = [(losses - baselines).detach(), torch.ones_like(losses, device=losses.device)]

    torch.autograd.backward(outputs, gradients)

    optimizer.step()

    return baselines.mean()


