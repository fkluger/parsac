import utils.metrics
import numpy as np
import sklearn.metrics


def calc_auc(error_array, cutoff=0.25):
    error_array = error_array.squeeze()
    error_array = np.sort(error_array)
    num_values = error_array.shape[0]

    plot_points = np.zeros((num_values, 2))

    midfraction = 1.

    for i in range(num_values):
        fraction = (i + 1) * 1.0 / num_values
        value = error_array[i]
        plot_points[i, 1] = fraction
        plot_points[i, 0] = value
        if i > 0:
            lastvalue = error_array[i - 1]
            if lastvalue < cutoff < value:
                midfraction = (lastvalue * plot_points[i - 1, 1] + value * fraction) / (value + lastvalue)

    if plot_points[-1, 0] < cutoff:
        plot_points = np.vstack([plot_points, np.array([cutoff, 1])])
    else:
        plot_points = np.vstack([plot_points, np.array([cutoff, midfraction])])

    sorting = np.argsort(plot_points[:, 0])
    plot_points = plot_points[sorting, :]

    auc = sklearn.metrics.auc(plot_points[plot_points[:, 0] <= cutoff, 0],
                              plot_points[plot_points[:, 0] <= cutoff, 1])
    auc = auc / cutoff

    return auc, plot_points


def compute_validation_metrics(opt, metrics, models, counts_total, gt_models, gt_labels, X, image_size, clusters, run_id, inv_intrinsics, train=False):
    if not opt.eval:
        if not ("inlier_count" in metrics.keys()):
            metrics["inlier_count"] = []

        metrics["inlier_count"] += [counts_total.mean().item()]

    if opt.problem == "vp":

        B, K, M, _, D = models.size()

        if not ("vp_error" in metrics.keys()):
            metrics["vp_error"] = [[] for _ in range(opt.runcount)]

        _, min_costs = utils.metrics.vp_loss(gt_models, models, inv_intrinsics, max_error=None)

        for bi in range(B):
            for ki in range(K):
                vps_true = gt_models[bi].cpu().detach().numpy()
                n = np.linalg.norm(vps_true, axis=-1)
                num_true = np.sum(n > 1e-8)

                errors = min_costs[bi, ki, 0, 0, :num_true].cpu().detach().numpy().tolist()

                metrics["vp_error"][run_id] += errors

    if (opt.dataset == "adelaide" or opt.dataset == "hope" or opt.dataset == "smh") and not train:

        if not ("misclassification_error" in metrics.keys()):
            metrics["misclassification_error"] = []
        if not ("geometric_error" in metrics.keys()):
            metrics["geometric_error"] = []

        ge = utils.metrics.geometric_errors(opt, X, image_size, gt_labels, models[..., 0, :])
        metrics["geometric_error"] += ge.flatten().tolist()

        classification_losses = utils.metrics.classification_loss(opt, gt_labels, clusters) * 100.0

        misclassification_errors = classification_losses.view(-1).cpu().numpy().tolist()
        metrics["misclassification_error"] += misclassification_errors

    return metrics


