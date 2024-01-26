import torch
import utils.solvers


def sample_minimal_sets(opt, log_probs):

    # batch x N x instances
    probs = torch.softmax(log_probs, dim=1)
    B, N, Mo = probs.size()
    M = opt.instances

    choice_weights = probs[..., :M].view(B, 1, 1, N, M).expand(B, opt.samplecount, opt.hypotheses, N, M)
    choice_weights = choice_weights.transpose(-1, -2).contiguous().view(-1, N)
    choice_batched = torch.multinomial(choice_weights, opt.mss, replacement=True)
    choices = choice_batched.view(B, opt.samplecount, opt.hypotheses, M, opt.mss)

    return choices


def generate_hypotheses(opt, X, choices):

    B, N, C = X.size()
    _, K, S, M, mss = choices.size()

    X_e = X.view(B, 1, 1, 1, N, C).expand(B, K, S, M, N, C)
    choices_e = choices.view(B, K, S, M, mss, 1).expand(B, K, S, M, mss, C)

    X_samples = torch.gather(X_e, -2, choices_e)

    hypotheses = utils.solvers.minimal_solver[opt.problem](X_samples)

    return hypotheses


def sample_hypotheses(opt, mode, hypotheses, weighted_inlier_counts, inlier_scores, residuals):
    B, K, S, M = weighted_inlier_counts.size()

    softmax_input = opt.softmax_alpha * weighted_inlier_counts

    hyp_selection_weights = torch.softmax(softmax_input, dim = 2)
    log_p_h_S = torch.nn.functional.log_softmax(softmax_input, dim=2)  # B, K, S, M

    choice_weights = hyp_selection_weights.transpose(-1, -2).contiguous().view(-1, S)

    if opt.hypsamples > 0 and mode == "train":
        choice_batched = torch.multinomial(choice_weights, opt.hypsamples, replacement=True)
        choices = choice_batched.view(B, K, M, opt.hypsamples)
        H = opt.hypsamples
    else:
        choice_batched = torch.argmax(choice_weights, dim=-1)
        choices = choice_batched.view(B, K, M, 1)
        H = 1

    hyp_choices_e = choices.view(B, K, 1, M, H)
    log_p_e = log_p_h_S.view(B, K, S, M, 1).expand(B, K, S, M, H)
    selected_log_p = torch.gather(log_p_e, 2, hyp_choices_e).squeeze(2)  # B, K, M, H
    log_p_M_S = selected_log_p.sum(2)  # B, K, H

    B, K, S, M, D = hypotheses.size()
    B, K, M, H = choices.size()
    B, K, S, M, N = inlier_scores.size()
    hypotheses_e = hypotheses.view(B, K, S, M, 1, D).expand(B, K, S, M, H, D)
    inlier_scores_e = inlier_scores.view(B, K, S, M, 1, N).expand(B, K, S, M, H, N)
    residuals_e = residuals.view(B, K, S, M, 1, N).expand(B, K, S, M, H, N)
    hyp_choices_e = choices.view(B, K, 1, M, H, 1).expand(B, K, 1, M, H, D)
    selected_hypotheses = torch.gather(hypotheses_e, 2, hyp_choices_e).squeeze(2)
    hyp_choices_e = choices.view(B, K, 1, M, H, 1).expand(B, K, 1, M, H, N)
    selected_inlier_scores = torch.gather(inlier_scores_e, 2, hyp_choices_e).squeeze(2)  # B, K, M, H, N
    selected_residuals = torch.gather(residuals_e, 2, hyp_choices_e).squeeze(2)

    return log_p_M_S, selected_inlier_scores, selected_hypotheses, selected_residuals


