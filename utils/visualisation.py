import matplotlib.collections
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torch
import numpy as np
import wandb
import utils.evaluation


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def generate_plot_single_batch(opt, X, log_inlier_weights, log_sample_weights, inlier_scores, clusters, labels,
                               gt_models, gt_labels, image, dataset, max_num_samples=6, metrics=None):
    B, N, Mo = log_inlier_weights.size()
    M = opt.instances
    inlier_weights = torch.softmax(log_inlier_weights, dim=2)
    sample_weights = torch.softmax(log_sample_weights, dim=1)
    inlier_weights = inlier_weights / torch.max(inlier_weights, dim=1, keepdim=True)[0]
    sample_weights = sample_weights / torch.max(sample_weights, dim=1, keepdim=True)[0]

    batch = min(B, max_num_samples)

    num_cols = 4 if opt.problem == "vp" else 5

    if True:
        fig, all_axes = plt.subplots(nrows=Mo, ncols=batch * num_cols, figsize=(batch * 4 * num_cols, Mo * 4))
        canvas = FigureCanvas(fig)
        plot_funs = {"vp": plot_vps, "homography": plot_fh, "fundamental": plot_fh}

        for mi, axes in enumerate(all_axes):
            for ai, ax in enumerate(axes):
                bi = int(ai / num_cols)

                x = X[bi].cpu().detach().numpy()

                size = 3

                if ai % num_cols == 0:
                    w = inlier_weights[bi, :, mi].cpu().detach().numpy()
                    c = '#1E88E5'
                    if opt.problem == "vp":
                        ax.set_facecolor('k')
                elif ai % num_cols == 1:
                    w = sample_weights[bi, :, mi].cpu().detach().numpy()
                    c = '#FFC107'
                    if opt.problem == "vp":
                        ax.set_facecolor('k')
                elif ai % num_cols == 2:
                    if mi < M:
                        w = inlier_scores[bi, mi].cpu().detach().numpy()
                    else:
                        if inlier_scores.size(1) == Mo:
                            w = inlier_scores[bi, 0].cpu().detach().numpy()
                        else:
                            w = 0
                    c = '#D81B60'

                    if opt.problem == "vp":
                        ax.set_facecolor('k')
                elif ai % num_cols == 3:
                    w = clusters[bi, (mi + 1) % Mo].cpu().detach().numpy()

                    c = '#D81B60'

                    if opt.problem == "vp":
                        ax.set_facecolor('k')
                elif ai % num_cols == 4:
                    if mi < M:
                        w = torch.where(gt_labels == (mi + 1),
                                        torch.ones(N, dtype=torch.float32, device=inlier_weights.device),
                                        torch.zeros(N, dtype=torch.float32,
                                                    device=inlier_weights.device)).cpu().detach().numpy()
                    else:
                        w = torch.where(gt_labels == 0,
                                        torch.ones(N, dtype=torch.float32, device=inlier_weights.device),
                                        torch.zeros(N, dtype=torch.float32,
                                                    device=inlier_weights.device)).cpu().detach().numpy()
                    c = 'g'
                else:
                    pass

                error = None
                if mi == 0 and ai == 0:
                    try:
                        error = metrics["misclassification_error"][-1]
                    except:
                        pass

                plot_funs[opt.problem](ax, x, w, c, error, size)

                ax.set_xticklabels([])
                ax.set_xticks([])
                ax.set_yticklabels([])
                ax.set_yticks([])
        plt.tight_layout()
        canvas.draw()
        plot_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close()

    if not opt.dataset == "adelaide":

        labels_ = torch.clamp(labels.transpose(-1, -2) - 1, min=0)
        probs = torch.softmax(log_sample_weights, dim=1)
        probs = probs / torch.max(probs, dim=1)[0][:, None, :]
        sample_weights = torch.gather(probs[0], 1, labels_).detach().cpu().numpy()[:, 0]
        probs = torch.softmax(log_inlier_weights, dim=2)
        probs = probs / torch.max(probs, dim=2)[0][..., None]
        inlier_weights = torch.gather(probs[0], 1, labels_).detach().cpu().numpy()[:, 0]

        labels_ = labels - 1

        p1, p2 = dataset.denormalise(X)
        p1 = p1.cpu().numpy()[0]
        p2 = p2.cpu().numpy()[0]

        fig, axs = plt.subplots(nrows=1, ncols=5, figsize=(5 * 4, 4), )
        canvas = FigureCanvas(fig)

        for ax_id in [0, 2, 3, 4]:
            axs[ax_id].set_facecolor('k')

        image_rgb = image[0].cpu().numpy()
        image_gray = rgb2gray(image_rgb)

        axs[0].imshow(image_rgb / 255.0)

        for ax in [axs[4]]:
            img = (image_gray.astype(float) / 255.0) * 0.7 + 0.3
            ax.imshow(img, cmap="Greys_r", vmin=0)
        img = np.zeros_like(img)
        axs[2].imshow(img, cmap="Greys_r", vmin=0, vmax=1)
        axs[3].imshow(img, cmap="Greys_r", vmin=0, vmax=1)

        colors_sw = []
        colors_iw = []
        colors_lb = []
        hues = [338, 208, 45, 170, 99, 310, 255, 80, 190, 230, 120]
        lws = []
        for ni in range(N):
            label = labels_[0, ni].item()
            if label == -1:
                hue = 0
                sat = 1
                val = 0
                val2 = 0
                lw = 0.1
            else:
                hue = hues[label % len(hues)] / 360.0
                sat = 1
                val = inlier_weights[ni]
                val2 = sample_weights[ni]
                lw = 4

            lws += [lw]
            color = matplotlib.colors.hsv_to_rgb([hue, sat, val])
            colors_iw += [color]
            color = matplotlib.colors.hsv_to_rgb([hue, sat, val2])
            colors_sw += [color]
            color = matplotlib.colors.hsv_to_rgb([hue, sat, 0 if label == -1 else 1])
            colors_lb += [color]

        if opt.problem == "vp":

            lines = [((a[0], a[1]), (b[0], b[1])) for a, b, in zip(p1.tolist(), p2.tolist())]
            lines2 = [((a[0], -a[1]), (b[0], -b[1])) for a, b, in zip(p1.tolist(), p2.tolist())]

            axs[1].set_xlim(0, 512)
            axs[1].set_ylim(-512, 0)

            lw = 3
            lc1 = matplotlib.collections.LineCollection(lines2, linewidths=lw, colors='k')
            lc2 = matplotlib.collections.LineCollection(lines, linewidths=lw, colors=colors_sw)
            lc3 = matplotlib.collections.LineCollection(lines, linewidths=lw, colors=colors_iw)
            lc4 = matplotlib.collections.LineCollection(lines, linewidths=lws, colors=colors_lb)

            axs[1].add_collection(lc1)
            axs[2].add_collection(lc2)
            axs[3].add_collection(lc3)
            axs[4].add_collection(lc4)
        else:
            lw = 4
            axs[1].scatter(p1[:, 0], p1[:, 1], s=lw, c='k')
            lw = 4
            axs[2].scatter(p1[:, 0], p1[:, 1], s=lw, c=colors_sw)
            axs[3].scatter(p1[:, 0], p1[:, 1], s=lw, c=colors_iw)
            axs[4].scatter(p1[:, 0], p1[:, 1], s=lws, c=colors_lb)

            for ax in axs:
                ax.set_xlim(np.min(p1) - 10, np.max(p1) + 10)
                ax.set_ylim(np.max(p1) + 10, np.min(p1) - 10)

        for ax in axs:
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.set_yticklabels([])
            ax.set_yticks([])
        plt.tight_layout()
        canvas.draw()
        plot_image2 = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        plot_image2 = plot_image2.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()

    return plot_image, plot_image2


def plot_fh(ax, X, weights, color, error, size):
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')

    ax.scatter(X[:, 0], X[:, 1], s=1, c='k')
    ax.scatter(X[:, 0], X[:, 1], s=weights * 20, c=color)

    if error is not None:
        ax.set_title("%.1f" % error)


def plot_vps(ax, X, weights, color, metrics, size):

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect('equal')

    p1 = X[:, 0:3]
    p2 = X[:, 3:6]
    p1 = p1 / p1[:, 2][:, None]
    p2 = p2 / p2[:, 2][:, None]

    lines = [((a[0], -a[1]), (b[0], -b[1])) for a, b, in zip(p1.tolist(), p2.tolist())]

    if weights is None:
        colors = 'k'
    else:
        cmap = plt.get_cmap('bone')
        colors = cmap(weights)

    lc1 = matplotlib.collections.LineCollection(lines, linewidths=size, colors=colors)
    ax.add_collection(lc1)


def save_visualisation_plots(opt, X, ranked_choices, log_probs_m, log_probs_n, inlier_scores, clusters, labels,
                             gt_models, gt_labels, image, dataset, **kwargs):
    N = inlier_scores.size(-1)
    B, K, M, H = ranked_choices.size()

    log_probs_e = log_probs_m.transpose(-1, -2).contiguous().view(B, 1, M + 1, 1, N).expand(B, K, M + 1, H, N)
    ranked_log_probs_m = torch.gather(log_probs_e, 2, ranked_choices.view(B, K, M, H, 1).expand(B, K, M, H, N))
    ranked_log_probs_m = torch.concatenate([ranked_log_probs_m, log_probs_e[..., -1, :, :].view(B, K, 1, H, N)], dim=2)
    log_probs_e = log_probs_n.transpose(-1, -2).contiguous().view(B, 1, M + 1, 1, N).expand(B, K, M + 1, H, N)
    ranked_log_probs_n = torch.gather(log_probs_e, 2, ranked_choices.view(B, K, M, H, 1).expand(B, K, M, H, N))
    ranked_log_probs_n = torch.concatenate([ranked_log_probs_n, log_probs_e[..., -1, :, :].view(B, K, 1, H, N)], dim=2)

    plot_image, plot_image2 = generate_plot_single_batch(opt, X, ranked_log_probs_m[:, 0, :, 0].transpose(-1, -2),
                                            ranked_log_probs_n[:, 0, :, 0].transpose(-1, -2), inlier_scores[:, 0, :, 0],
                                            clusters[:, 0, 0], labels[:, 0, 0], gt_models, gt_labels, image, dataset,
                                            max_num_samples=6, **kwargs)

    assert False, "TODO: save plots to somewhere"



def log_wandb(log_data, metrics, mode, epoch, images=[]):
    print_metrics = ["loss", "time", "total_time"]

    print_string = " | ".join([("%s: %.3f" % (key, np.mean(metrics[key]))) for key in print_metrics])
    print(print_string)

    for key, value in metrics.items():
        if "vp_err" in key:
            for cutoff in (1, 3, 5, 10):
                aucs = []
                for errors in value:
                    try:
                        auc, plot_points = utils.evaluation.calc_auc(np.array(errors), cutoff=cutoff)
                    except:
                        auc = 0
                    aucs += [auc]
                log_data["%s/%s_auc_%d_avg" % (mode, key, cutoff)] = np.mean(aucs)
                log_data["%s/%s_auc_%d_std" % (mode, key, cutoff)] = np.std(aucs)

            log_data["%s/%s_avg" % (mode, key)] = np.mean(value)
            log_data["%s/%s_std" % (mode, key)] = np.std(value)
            log_data["%s/%s_med" % (mode, key)] = np.median(value)
        else:
            try:
                log_data["%s/%s_avg" % (mode, key)] = np.mean(value)
                log_data["%s/%s_std" % (mode, key)] = np.std(value)
                log_data["%s/%s_med" % (mode, key)] = np.median(value)
            except:
                pass

    wandb.log(log_data, step=epoch)
    if len(images) > 0:
        for idx, wandb_image in enumerate(images):
            wandb.log({"%s/examples" % mode: wandb_image}, step=idx + 1)
