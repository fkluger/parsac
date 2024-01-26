from networks.cn_net import CNNet
from utils.tee import Tee
from datasets.nyuvp import NYUVP
from datasets.su3 import SU3
from datasets.adelaide import AdelaideRMFDataset
from datasets.hope import HopeFDataset
from datasets.smh import SMHDataset
import torch.optim as optim
import torch
import os
import json
import wandb
import numpy as np
import random


def seeds(opt):
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_logging_and_checkpointing(opt):
    wandb.init(project="parsac", entity=opt.wandb_entity, group=opt.wandb_group, mode=opt.wandb, dir=opt.wandb_dir)

    parent_ckpt_dir = os.path.join(opt.checkpoint_dir, opt.wandb_group)
    os.makedirs(parent_ckpt_dir, exist_ok=True)

    dir_success = False

    while not dir_success:
        if opt.debug_session:
            ckpt_dir = os.path.join(opt.checkpoint_dir, opt.group, "debug_session")
            os.makedirs(ckpt_dir, exist_ok=True)
            dir_success = True
        else:
            run_name = wandb.run.name
            if run_name is None or run_name == "":
                if not (opt.jobid == ""):
                    run_name = opt.jobid
                else:
                    run_name = "%08d" % np.random.randint(0, 99999999)
            ckpt_dir = os.path.join(opt.checkpoint_dir, opt.wandb_group, run_name)
            try:
                os.makedirs(ckpt_dir, exist_ok=False)
                dir_success = True
            except FileExistsError as err:
                print(err)
                print("%s exists, try again.." % ckpt_dir)

    print("saving models to: ", ckpt_dir)

    log_file = os.path.join(ckpt_dir, "output.log")
    print("log file: ", log_file)
    log = Tee(log_file, "w", file_only=False)

    with open(os.path.join(ckpt_dir, 'commandline_args.txt'), 'w') as f:
        json.dump(opt.__dict__, f, indent=2)

    wandb.config.update(vars(opt))
    wandb.config.update({"checkpoints": ckpt_dir})

    return ckpt_dir, log


def get_model(opt):
    device_id = int(opt.gpu)
    if device_id is None or device_id < 0 or not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', device_id)

    input_dim_map = {
        "nyuvp": 4,
        "yudplus": 4,
        "yud": 4,
        "su3": 4,
        "adelaide": 4,
        "hope": 4,
        "smh": 4,
    }

    input_dim = input_dim_map[opt.dataset]

    model = CNNet(input_dim, opt.instances+1, opt.network_layers, batch_norm=True,
                  separate_weights=opt.separate_weights)

    if opt.load is not None and len(opt.load) > 0:
        print("load weights from ", opt.load)
        model.load_state_dict(torch.load(os.path.join(opt.load, "model_weights.net"), map_location=device), strict=True)

    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, eps=1e-4, weight_decay=1e-4)
    if opt.load is not None and len(opt.load) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(opt.load, "optimizer.net"), map_location=device))

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_steps, gamma=0.1)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of network parameters: {total_params}")

    return model, optimizer, scheduler, device


def get_dataset(opt):
    test_dataset = None
    val_dataset = None
    train_dataset = None

    return_images = opt.visualise and not ("train" in opt.modes)
    cache = True

    if opt.dataset == "nyuvp":
        train_dataset = NYUVP("train", opt.max_num_points)
        val_dataset = NYUVP("val", opt.max_num_points)
        test_dataset = NYUVP("test", opt.max_num_points, deeplsd_folder=opt.ablation_deeplsd_folder, cache=False)
    elif opt.dataset == "su3":
        train_dataset = SU3(opt.data_path, "train", opt.max_num_points, normalise_coords=True, augmentation=opt.augment,
                            deeplsd_folder=opt.ablation_deeplsd_folder)
        val_dataset = SU3(opt.data_path, "valid", opt.max_num_points, normalise_coords=True,
                          deeplsd_folder=opt.ablation_deeplsd_folder)
        test_dataset = SU3(opt.data_path, "test", opt.max_num_points, normalise_coords=True, cache=True,
                           deeplsd_folder=opt.ablation_deeplsd_folder, ablation_outlier_ratio=opt.ablation_outlier_ratio,
                           ablation_noise=opt.ablation_noise)
    elif opt.dataset == "yudplus" or opt.dataset == "yud":
        train_dataset = NYUVP("train", opt.max_num_points, use_yud=True, use_yud_plus=(opt.dataset == "yudplus"))
        test_dataset = NYUVP("test", opt.max_num_points, use_yud=True, use_yud_plus=(opt.dataset == "yudplus"),
                             deeplsd_folder=opt.ablation_deeplsd_folder, cache=False)
    elif opt.dataset == "adelaide":
        test_dataset = AdelaideRMFDataset(opt.data_path, opt.max_num_points, problem=opt.problem, permute_points=False,
                                          ablation_outlier_ratio=opt.ablation_outlier_ratio,
                                          ablation_noise=opt.ablation_noise)
    elif opt.dataset == "hope":
        train_dataset = HopeFDataset(opt.data_path, "train", opt.max_num_points)
        val_dataset = HopeFDataset(opt.data_path, "val", opt.max_num_points, return_images=return_images)
        test_dataset = HopeFDataset(opt.data_path, "test", opt.max_num_points, return_images=return_images,
                                    cache_data=cache)
    elif opt.dataset == "smh":
        train_dataset = SMHDataset(opt.data_path, "train", opt.max_num_points, keep_in_mem=cache)
        val_dataset = SMHDataset(opt.data_path, "val", opt.max_num_points, keep_in_mem=cache,
                                 return_images=return_images)
        test_dataset = SMHDataset(opt.data_path, "test", opt.max_num_points, keep_in_mem=cache,
                                  return_images=return_images)
    else:
        assert False, "unknown dataset %s" % opt.dataset

    if opt.dataset == "nyuvp":
        fx_rgb = 5.1885790117450188e+02
        fy_rgb = 5.1946961112127485e+02
        cx_rgb = 3.2558244941119034e+02
        cy_rgb = 2.5373616633400465e+02
        intrinsics = torch.tensor([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
        scale_matrix = torch.tensor([[1. / 320., 0, -1.], [0, 1. / 320., -.75], [0, 0, 1]])
        SKmat = scale_matrix @ intrinsics
        inv_intrinsics = torch.inverse(SKmat)
    elif opt.dataset == "yudplus" or opt.dataset == "yud":
        f = 6.053170589753693
        ps = 0.00896875
        pp = [307.55130528, 251.45424496]
        intrinsics = torch.tensor([[f / ps, 0, pp[0]], [0, f / ps, pp[1]], [0, 0, 1]])
        scale_matrix = torch.tensor([[1. / 320., 0, -1.], [0, 1. / 320., -.75], [0, 0, 1]])
        SKmat = scale_matrix @ intrinsics
        inv_intrinsics = torch.inverse(SKmat)
    elif opt.dataset == "su3":
        f = 2.1875 * 256
        c = 256
        intrinsics = torch.tensor([[f, 0, c], [0, -f, c], [0, 0, 1]])
        scale_matrix = torch.tensor([[1. / c, 0, -1.], [0, 1. / c, -1.], [0, 0, 1]])
        SKmat = scale_matrix @ intrinsics
        inv_intrinsics = torch.inverse(SKmat)
    else:
       inv_intrinsics = None

    print("initialised %s dataset" % opt.dataset)

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset, "inverse_intrinsics": inv_intrinsics}


def get_dataloader(opt, datasets, shuffle_all=False):
    g = torch.Generator()
    g.manual_seed(opt.seed)

    dataloaders = {}
    for split, dataset in datasets.items():
        if dataset is None:
            loader = None
        else:
            loader = torch.utils.data.DataLoader(dataset, shuffle=(split == "train" or shuffle_all),
                                                 num_workers=opt.num_workers, generator=g, worker_init_fn=seed_worker,
                                                 batch_size=opt.batch, drop_last=False)
        dataloaders[split] = loader
    return dataloaders


def get_parameter_sweep_range(opt):

    if opt.parameter_sweep == "S":
        values = range(opt.hypotheses//8, opt.hypotheses*2+1, opt.hypotheses//8)
    elif opt.parameter_sweep == "tau":
        if opt.problem == "homography" and opt.dataset == "adelaide":
            values = np.logspace(np.log10(opt.inlier_threshold)-3, np.log10(opt.inlier_threshold)+1, num=100, endpoint=True)
        elif opt.problem == "homography" and opt.dataset == "smh":
            values = np.logspace(np.log10(opt.inlier_threshold)+1, np.log10(opt.inlier_threshold)+3, num=50, endpoint=True)
        else:
            values = np.logspace(np.log10(opt.inlier_threshold)-1, np.log10(opt.inlier_threshold)+1, num=50, endpoint=True)
    elif opt.parameter_sweep == "taua":
        if opt.problem == "homography" and opt.dataset == "adelaide":
            values = np.logspace(np.log10(opt.assignment_threshold)-4, np.log10(opt.assignment_threshold)+1, num=150, endpoint=True)
        elif opt.problem == "homography" and opt.dataset == "smh":
            values = np.logspace(np.log10(opt.assignment_threshold)+3, np.log10(opt.assignment_threshold)+4, num=25, endpoint=True)
        else:
            values = np.logspace(np.log10(opt.assignment_threshold)-1, np.log10(opt.assignment_threshold)+1, num=50, endpoint=True)
    elif opt.parameter_sweep == "beta":
        values = np.logspace(np.log10(opt.inlier_softness)-1, np.log10(opt.inlier_softness)+1, num=50, endpoint=True)
    else:
        assert False

    if opt.dataset in ["hope", "smh", "nyuvp", "su3"]:
        opt.modes = ["val"]
    else:
        opt.modes = ["test"]

    return values