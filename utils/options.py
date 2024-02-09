import argparse


def get_options():
    parser = argparse.ArgumentParser(
        description='PARSAC: Accelerating Robust Multi-Model Fitting with Parallel Sample Consensus',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eval', dest='eval', action='store_true', help='set some default values to run evaluation only')
    # general:
    parser.add_argument('--problem', default="vp", choices=["vp", "homography", "fundamental"], help='type of problem')
    parser.add_argument('--dataset', default="nyuvp",
                        choices=["nyuvp", "yudplus", "yud", "su3", "adelaide", "hope", "smh"],
                        help='name of dataset to use')
    parser.add_argument('--data_path', default="", help='path to dataset')
    parser.add_argument('--modes', nargs="+", default=["val", "train"], choices=["train", "val", "test"])
    parser.add_argument('--checkpoint_dir', default='./tmp/checkpoints',
                        help='directory for storing neural network weight checkpoints')
    parser.add_argument('--load', default="", type=str, help='load pretrained neural network weights from folder')
    parser.add_argument('--gpu', default="0", help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--jobid', default="", help='Custom job ID (for setting the checkpoint directory)')

    # hyper-parameters:
    parser.add_argument('--instances', type=int, default=4, help='M-hat - number of putative model instances')
    parser.add_argument('--inlier_threshold', type=float, default=1e-4, help='tau - inlier threshold')
    parser.add_argument('--inlier_softness', default=5, type=float, help='beta - inlier softness')
    parser.add_argument('--assignment_threshold', type=float, default=1e-2, help='tau_a - assignment threshold')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training')
    parser.add_argument('--softmax_alpha', type=float, default=1000., help='alpha_s - softmax scale factor')
    parser.add_argument('--epochs', type=int, default=2000, help='N_e - number of training epochs')
    parser.add_argument('--lr_steps', nargs="+", default=[1500],
                        help='N_lr - number of epochs before reducing LR by factor of 10')
    parser.add_argument('--batch', type=int, default=48, help='B - batch size')
    parser.add_argument('--samplecount', type=int, default=1, help='K - hypotheses set samples')
    parser.add_argument('--hypsamples', type=int, default=0,
                        help='K_hat - model set samples (0 = select argmax of inlier count)')
    parser.add_argument('--hypotheses', type=int, default=4, help='S - number of model hypotheses')
    parser.add_argument('--max_num_points', default=512, type=int,
                        help='|X| - max. number of observations to select for input (set to -1 in order to use all observations)')
    parser.add_argument('--network_layers', default=6, type=int, help='number of residual blocks')

    # training only:
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data loader')
    parser.add_argument('--ckpt_mode', default="last", type=str,
                        help='', choices=["all", "last", "disabled"])
    parser.add_argument('--self_supervised', action='store_true', help='use self-supervised loss function')
    parser.add_argument('--cumulative_loss', type=float, default=0.3, help='gamma - weight for self-supervised loss')

    # evaluation only:
    parser.add_argument('--runcount', type=int, default=1, help='perform multiple runs and compute mean and standard deviation of all metrics')

    # ablation
    parser.add_argument('--ablation_outlier_ratio', type=float, default=-1, help='synthetic outlier ratio')
    parser.add_argument('--ablation_noise', type=float, default=0, help='gaussian noise sigma')
    parser.add_argument('--ablation_deeplsd_folder',
                        default=None, type=str, help='folder with pre-computed DeepLSD features')
    parser.add_argument('--inlier_counting', default="weighted", type=str,
                        choices=["weighted", "unweighted"], help="use weighted or unweighted inlier counting")
    parser.add_argument('--inlier_function', default="soft", choices=["hard", "soft"],
                        help='use hard or soft inlier scoring function')
    parser.add_argument('--no_refine', dest='refine', action='store_false',
                        help='disable refinement for vanishing points')
    parser.add_argument('--no_separate_weights', dest='separate_weights', action='store_false',
                        help="only predict one set of log weights for sample and inlier weights")

    # other
    parser.add_argument('--mss', default=0, type=int,
                        help='minimal set size (overrides value given by the problem type)')
    parser.add_argument('--debug_session', action='store_true', help='')
    parser.add_argument('--augment', action='store_true', help='')
    parser.add_argument('--visualise', action='store_true', help='')

    # logging
    parser.add_argument('--wandb_group', default="", type=str, help='Weights and Biases group')
    parser.add_argument('--wandb', default="disabled", choices=["online", "offline", "disabled"],
                        help='Weights and Biases mode')
    parser.add_argument('--wandb_entity', default="tnt", help='Weights and Biases entity')
    parser.add_argument('--wandb_dir', default="./tmp", type=str,
                        help='Weights and Biases offline storage folder')

    opt = parser.parse_args()

    if opt.mss < 2:
        mss_map = {"lines": 2, "vp": 2, "homography": 4, "fundamental": 7}
        opt.mss = mss_map[opt.problem]

    if opt.eval:
        opt.modes = ["test"]
        opt.batch = 1
        opt.max_num_points = -1
        opt.num_workers = 0
        opt.epochs = 1
        opt.hypsamples = 0
        opt.samplecount = 1
        opt.wandb = "disabled"
        opt.runcount = 5

    return opt
