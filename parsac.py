from utils import \
    options, initialisation, sampling, backward, visualisation, evaluation, residual_functions, inlier_counting, metrics, postprocessing
import torch
import time

opt = options.get_options()

initialisation.seeds(opt)

ckpt_dir, log = initialisation.setup_logging_and_checkpointing(opt)

model, optimizer, scheduler, device = initialisation.get_model(opt)

datasets = initialisation.get_dataset(opt)

for epoch in range(opt.epochs):

    print("epoch %d / %d" % (epoch + 1, opt.epochs))

    dataloaders = initialisation.get_dataloader(opt, datasets, shuffle_all=False)

    for mode in opt.modes:

        assert not (dataloaders[mode] is None), "no dataloader for %s available" % mode

        print("mode: %s" % mode)

        if mode == "train":
            model.train()
        else:
            model.eval()

        eval_metrics = {"loss": [], "time": [], "total_time": []}
        wandb_log_data = {}

        total_start = time.time()

        for batch_idx, (features, X, gt_labels, gt_models, image, image_size) in enumerate(dataloaders[mode]):

            for run_idx in range(opt.runcount):

                X = X.to(device)
                features = features.to(device)
                gt_labels = gt_labels.to(device)
                gt_models = gt_models.to(device)
                image_size = image_size.to(device)

                optimizer.zero_grad()

                start_time = time.time()

                log_inlier_weights, log_sample_weights = model(features)

                with torch.no_grad():

                    minimal_sets = sampling.sample_minimal_sets(opt, log_sample_weights)
                    hypotheses = sampling.generate_hypotheses(opt, X, minimal_sets)
                    residuals = residual_functions.compute_residuals(opt, X, hypotheses)

                    weighted_inlier_ratios, inlier_scores = \
                        inlier_counting.count_inliers(opt, residuals, log_inlier_weights)

                log_p_M_S, sampled_inlier_scores, sampled_hypotheses, sampled_residuals = \
                    sampling.sample_hypotheses(opt, mode, hypotheses, weighted_inlier_ratios, inlier_scores, residuals)

                if opt.refine:
                    if opt.problem == "vp":
                        sampled_hypotheses, sampled_residuals, sampled_inlier_scores = \
                            postprocessing.refinement_with_inliers(opt, X, sampled_inlier_scores)

                ranked_choices, ranked_inlier_ratios, ranked_hypotheses, ranked_scores, labels, clusters = \
                    postprocessing.ranking_and_clustering(opt, sampled_inlier_scores, sampled_hypotheses,
                                                          sampled_residuals)

                duration = (time.time() - start_time) * 1000

                eval_metrics["time"] += [duration]

                if not opt.self_supervised:
                    with torch.no_grad():
                        if opt.problem == "vp":
                            exp_losses, _ = metrics.vp_loss(gt_models, ranked_hypotheses, datasets["inverse_intrinsics"])
                        elif opt.problem == "fundamental" or opt.problem == "homography":
                            exp_losses = metrics.classification_loss(opt, gt_labels, clusters)
                        else:
                            assert False
                else:
                    cumulative_inlier_losses = inlier_counting.compute_cumulative_inliers(opt, ranked_scores)
                    sample_inlier_counts = inlier_counting.combine_hypotheses_inliers(ranked_scores)
                    exp_losses = backward.expected_self_losses(opt, sample_inlier_counts, cumulative_inlier_losses)

                if mode == "train":
                    log_p_M = backward.log_probabilities(log_sample_weights, minimal_sets, log_p_M_S)
                    _ = backward.backward_pass(opt, exp_losses, log_p_M, optimizer)
                else:
                    eval_metrics = \
                        evaluation.compute_validation_metrics(opt, eval_metrics, ranked_hypotheses,
                                                              ranked_inlier_ratios, gt_models, gt_labels, X, image_size, clusters,
                                                              run_idx, datasets["inverse_intrinsics"], train=(mode == "train"))
                mean_loss = exp_losses.mean()
                eval_metrics["loss"] += [mean_loss.item()]

                total_duration = (time.time() - total_start) * 1000
                eval_metrics["total_time"] += [total_duration]
                total_start = time.time()

                if opt.visualise:
                    visualisation.save_visualisation_plots(opt, X, ranked_choices, log_inlier_weights,
                                                           log_sample_weights, ranked_scores, clusters,
                                                           labels, gt_models, gt_labels, image,
                                                           dataloaders[mode].dataset, metrics=eval_metrics)

        visualisation.log_wandb(wandb_log_data, eval_metrics, mode, epoch)
        if opt.eval:
            for key, val in wandb_log_data.items():
                print(key, ":", val)

        if mode == "train":
            scheduler.step()

    if opt.ckpt_mode == "all":
        torch.save(model.state_dict(), '%s/model_weights_%06d.net' % (ckpt_dir, epoch))
        torch.save(optimizer.state_dict(), '%s/optimizer_%06d.net' % (ckpt_dir, epoch))
    elif opt.ckpt_mode == "last":
        torch.save(model.state_dict(), '%s/model_weights.net' % (ckpt_dir))
        torch.save(optimizer.state_dict(), '%s/optimizer.net' % (ckpt_dir))
