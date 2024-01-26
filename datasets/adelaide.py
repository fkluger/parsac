import os.path
import scipy.io
from torch.utils.data import Dataset
import numpy as np

class AdelaideRMF:

    homography_sequences = [
        "barrsmith.mat",
        "bonhall.mat",
        "bonython.mat",
        "elderhalla.mat",
        "elderhallb.mat",
        "hartley.mat",
        "johnsona.mat",
        "johnsonb.mat",
        "ladysymon.mat",
        "library.mat",
        "napiera.mat",
        "napierb.mat",
        "neem.mat",
        "nese.mat",
        "oldclassicswing.mat",
        "physics.mat",
        "sene.mat",
        "unihouse.mat",
        "unionhouse.mat",
    ]

    fundamental_sequences = [
        "cube.mat",
        "book.mat",
        "biscuit.mat",
        "game.mat",
        "biscuitbook.mat",
        "breadcube.mat",
        "breadtoy.mat",
        "cubechips.mat",
        "cubetoy.mat",
        "gamebiscuit.mat",
        "breadtoycar.mat",
        "carchipscube.mat",
        "toycubecar.mat",
        "breadcubechips.mat",
        "biscuitbookbox.mat",
        "cubebreadtoychips.mat",
        "breadcartoychips.mat",
        "dinobooks.mat",
        "boardgame.mat"
    ]

    def __init__(self, data_dir, keep_in_mem=True, normalize_coords=False, return_images=False, problem="homography",
                 ablation_outlier_ratio=-1, ablation_noise=0):

        self.data_dir = data_dir
        self.keep_in_mem = keep_in_mem
        self.normalize_coords = normalize_coords
        self.return_images = return_images
        self.ablation_outlier_ratio = ablation_outlier_ratio
        self.ablation_noise = ablation_noise

        self.dataset_files = []

        if problem == "homography":
            self.sequences = self.homography_sequences
        elif problem == "fundamental":
            self.sequences = self.fundamental_sequences
        elif problem == "all":
            self.sequences = self.fundamental_sequences + self.homography_sequences
        else:
            assert False

        for seq in self.sequences:
            seq_ = os.path.join(self.data_dir, seq)
            self.dataset_files += [seq_]

        self.dataset = [None for _ in self.dataset_files]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):

        filename = self.dataset_files[key]
        datum = self.dataset[key]

        if datum is None:

            data_mat = scipy.io.loadmat(filename, variable_names=["data", "label", "score", "img1", "img2"])
            pts1 = np.transpose(data_mat["data"][0:2,:])
            pts2 = np.transpose(data_mat["data"][3:5,:])

            if self.ablation_noise > 0:
                pts1 += np.random.normal(0, self.ablation_noise, pts1.shape)
                pts2 += np.random.normal(0, self.ablation_noise, pts2.shape)

            sideinfo = data_mat["score"].squeeze()
            gt_label = data_mat["label"].squeeze()
            img1size = data_mat["img1"].shape[0:2]
            img2size = data_mat["img2"].shape[0:2]

            if self.ablation_outlier_ratio >= 0:
                pts1 = pts1[np.where(gt_label > 0)]
                pts2 = pts2[np.where(gt_label > 0)]
                sideinfo = sideinfo[np.where(gt_label > 0)]
                gt_label = gt_label[np.where(gt_label > 0)]

                if self.ablation_outlier_ratio > 0:
                    N = pts1.shape[0]
                    No = int(N * self.ablation_outlier_ratio / (1 - self.ablation_outlier_ratio))

                    opts1 = np.zeros((No, 2)).astype(np.float32)
                    opts2 = np.zeros((No, 2)).astype(np.float32)
                    osideinfo = np.zeros(No).astype(np.float32)
                    ogt_label = np.zeros(No).astype(np.float32)
                    opts1[:, 0] = np.random.uniform(0, img1size[1]-1, No)
                    opts1[:, 1] = np.random.uniform(0, img1size[0]-1, No)
                    opts2[:, 0] = np.random.uniform(0, img2size[1]-1, No)
                    opts2[:, 1] = np.random.uniform(0, img2size[0]-1, No)

                    pts1 = np.concatenate([pts1, opts1], axis=0)
                    pts2 = np.concatenate([pts2, opts2], axis=0)
                    sideinfo = np.concatenate([sideinfo, osideinfo], axis=0)
                    gt_label = np.concatenate([gt_label, ogt_label], axis=0)

            if self.normalize_coords:
                scale1 = np.max(img1size)
                scale2 = np.max(img2size)

                pts1[:,0] -= img1size[1]/2.
                pts2[:,0] -= img2size[1]/2.
                pts1[:,1] -= img1size[0]/2.
                pts2[:,1] -= img2size[0]/2.
                pts1 /= (scale1/2.)
                pts2 /= (scale2/2.)

            datum = {'points1': pts1, 'points2': pts2, 'sideinfo': sideinfo, 'img1size': img1size, 'img2size': img2size,
                     'labels': gt_label, 'name': self.sequences[key][:-4]}

            if self.return_images:
                datum["img1"] = data_mat["img1"]
                datum["img2"] = data_mat["img2"]

            if self.keep_in_mem:
                self.dataset[key] = datum

        return datum


class AdelaideRMFDataset(Dataset):

    def __init__(self, data_dir_path, max_num_points, keep_in_mem=True, ablation_outlier_ratio=-1, ablation_noise=0,
                 permute_points=True, return_images=False, return_labels=True, problem="homography"):
        self.homdata = AdelaideRMF(data_dir_path, keep_in_mem, normalize_coords=True, return_images=return_images,
                                   problem=problem, ablation_outlier_ratio=ablation_outlier_ratio,
                                   ablation_noise=ablation_noise)
        self.max_num_points = max_num_points
        self.permute_points = permute_points
        self.return_images = return_images
        self.return_labels = return_labels

    def __len__(self):
        return len(self.homdata)

    def __getitem__(self, key):
        datum = self.homdata[key]

        if self.max_num_points <= 0:
            max_num_points = datum['points1'].shape[0]
        else:
            max_num_points = self.max_num_points

        if self.permute_points:
            perm = np.random.permutation(datum['points1'].shape[0])
            datum['points1'] = datum['points1'][perm]
            datum['points2'] = datum['points2'][perm]
            datum['sideinfo'] = datum['sideinfo'][perm]
            datum['labels'] = datum['labels'][perm]

        points = np.zeros((max_num_points, 5)).astype(np.float32)
        mask = np.zeros((max_num_points, )).astype(int)
        labels = np.zeros((max_num_points, )).astype(int)

        num_actual_points = np.minimum(datum['points1'].shape[0], max_num_points)
        points[0:num_actual_points, 0:2] = datum['points1'][0:num_actual_points, :].copy()
        points[0:num_actual_points, 2:4] = datum['points2'][0:num_actual_points, :].copy()
        points[0:num_actual_points, 4] = datum['sideinfo'][0:num_actual_points].copy()
        labels[0:num_actual_points] = datum['labels'][0:num_actual_points].copy()

        mask[0:num_actual_points] = 1

        imgsize = np.array([datum['img1size'], datum['img2size']])

        if num_actual_points < max_num_points:
            for i in range(num_actual_points, max_num_points, num_actual_points):
                rest = max_num_points-i
                num = min(rest, num_actual_points)
                points[i:i+num, :] = points[0:num, :].copy()
                labels[i:i+num] = labels[0:num].copy()

        return points, points, labels, 0, 0, imgsize


def make_vis():
    import matplotlib
    import matplotlib.pyplot as plt
    def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    dataset = AdelaideRMF("../datasets/adelaide", keep_in_mem=False, normalize_coords=False, return_images=True,
                          problem="fundamental", ablation_outlier_ratio=-1, ablation_noise=0)

    target_folder = "./tmp/fig/adelaide_f"

    os.makedirs(target_folder, exist_ok=True)

    for idx in range(len(dataset)):
        print("%d / %d" % (idx+1, len(dataset)), end="\r")
        sample = dataset[idx]
        img1 = sample["img1"].astype(np.uint8)
        img2 = sample["img2"].astype(np.uint8)
        pts1 = sample["points1"]
        pts2 = sample["points2"]
        y = sample["labels"]

        num_models = np.max(y)

        cb_hex = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#8e10b3",
                  "#374009", "#aec8ea", "#56611b", "#64a8c6", "#99d8d4", "#745a50", "#46fa50", "#e09eea", "#5b2b1f",
                  "#723f91", "#634418", "#7db0d0", "#1ae37c", "#aa462c", "#719bb7", "#463aa2", "#98f42e", "#32185b",
                  "#364fcd", "#7e54c8", "#bb5f7f", "#d466d5", "#5a0382", "#443067", "#a76232", "#78dbc1", "#35a4b2",
                  "#52d387", "#af5a7e", "#3ef57d", "#d6d993"]
        cb = np.array([matplotlib.colors.to_rgb(x) for x in cb_hex])

        fig = plt.figure(figsize=(4 * 4, 4 * 2), dpi=256)
        axs = fig.subplots(nrows=1, ncols=2)
        for ax in axs:
            ax.set_aspect('equal', 'box')
            ax.axis('off')

        if len(img1.shape) == 3:
            img1g = rgb2gray(img1) * 0.5 + 128
        else:
            img1g = img1 * 0.5 + 128

        if len(img2.shape) == 3:
            img2g = rgb2gray(img2) * 0.5 + 128
        else:
            img2g = img2 * 0.5 + 128

        axs[0].imshow(img1g, cmap='Greys_r', vmin=0, vmax=255)
        axs[1].imshow(img2g, cmap='Greys_r', vmin=0, vmax=255)

        for j, pts in enumerate([pts1, pts2]):
            ax = axs[j]

            ms = np.where(y>0, 8, 4)

            c = cb[y]

            ax.scatter(pts[:, 0], pts[:, 1], c=c, s=ms**2)

        fig.tight_layout()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(target_folder, "%02d_%03d_vis.png" % (num_models, idx)), bbox_inches='tight',
                    pad_inches=0)
        plt.close()

        fig = plt.figure(figsize=(4 * 4, 4 * 2), dpi=150)
        axs = fig.subplots(nrows=1, ncols=2)
        for ax in axs:
            ax.set_aspect('equal', 'box')
            ax.axis('off')

        if len(img1.shape) == 3:
            axs[0].imshow(img1)
        else:
            axs[0].imshow(img1, cmap="Greys_r")

        if len(img2.shape) == 3:
            axs[1].imshow(img2)
        else:
            axs[1].imshow(img2, cmap="Greys_r")

        fig.tight_layout()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(target_folder, "%02d_%03d_orig.png" % (num_models, idx)), bbox_inches='tight',
                    pad_inches=0)
        plt.close()
