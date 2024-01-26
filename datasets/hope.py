import os.path
from torch.utils.data import Dataset
import numpy as np
import pickle
import skimage
import random

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

class HopeF:

    def __init__(self, data_dir, split, cache_data=True, normalize_coords=True, return_images=False, return_gt_models=False):

        self.data_dir = data_dir
        self.cache_data = cache_data
        self.normalize_coords = normalize_coords
        self.return_images = return_images
        self.return_gt_models = return_gt_models

        if split == "train":
            data_ids = range(800)
        elif split == "val":
            data_ids = range(800, 900)
        elif split == "test":
            data_ids = range(900, 1000)
        elif split == "all":
            data_ids = range(0, 1000)
        else:
            assert False, "invalid split: %s" % split

        self.dataset_folders = []

        for num_obj in [1, 2, 3, 4]:
            self.dataset_folders += [os.path.join(self.data_dir, "%d/%04d" % (num_obj, i)) for i in data_ids]

        self.img_size = (1024, 1024)

        cache_folders = ["/phys/ssd/tmp/hope", "/phys/ssd/slurmstorage/tmp/hope", "/tmp/hope", "/phys/intern/tmp/hope"]

        self.cache_dir = None
        if cache_data:
            for cache_folder in cache_folders:
                try:
                    cache_folder = os.path.join(cache_folder, split)
                    os.makedirs(cache_folder, exist_ok=True)
                    self.cache_dir = cache_folder
                    print("%s is cache folder" % cache_folder)
                    break
                except:
                    print("%s unavailable" % cache_folder)

    def __len__(self):
        return len(self.dataset_folders)

    def __getitem__(self, key):

        folder = self.dataset_folders[key]
        datum = None

        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, "%09d.pkl" % key)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    datum = pickle.load(f)

        if datum is None:

            features_and_gt = np.load(os.path.join(folder, "features_and_ground_truth.npz"), allow_pickle=True)

            gt_label = features_and_gt["labels"]
            pts1 = features_and_gt["points1"][:, :2]
            pts2 = features_and_gt["points2"][:, :2]
            sideinfo = features_and_gt["ratios"]

            if self.normalize_coords:
                scale = np.max(self.img_size)

                pts1[:,0] -= self.img_size[1]/2.
                pts2[:,0] -= self.img_size[1]/2.
                pts1[:,1] -= self.img_size[0]/2.
                pts2[:,1] -= self.img_size[0]/2.
                pts1 /= (scale/2.)
                pts2 /= (scale/2.)

            datum = {'points1': pts1, 'points2': pts2, 'sideinfo': sideinfo, 'img1size': self.img_size,
                     'img2size': self.img_size, 'labels': gt_label}

            if self.cache_dir is not None:
                cache_path = os.path.join(self.cache_dir, "%09d.pkl" % key)
                if not os.path.exists(cache_path):
                    with open(cache_path, 'wb') as f:
                        pickle.dump(datum, f, pickle.HIGHEST_PROTOCOL)

        if self.return_images:
            img1_path = os.path.join(folder, "render0.png")
            img2_path = os.path.join(folder, "render1.png")
            image1_rgb = skimage.io.imread(img1_path).astype(float)[:, :, :3]
            image2_rgb = skimage.io.imread(img2_path).astype(float)[:, :, :3]
            image1 = rgb2gray(image1_rgb)
            image2 = rgb2gray(image2_rgb)
            datum['image1'] = image1
            datum['image2'] = image2
            datum['img1'] = image1_rgb
            datum['img2'] = image2_rgb

        if self.return_gt_models:
            features_and_gt = np.load(os.path.join(folder, "features_and_ground_truth.npz"), allow_pickle=True)
            gt = features_and_gt["F"]
            datum["gt"] = gt

        return datum


class HopeFDataset(Dataset):

    def __init__(self, data_dir_path, split, max_num_points, cache_data=True, normalize_coords=True,
                 permute_points=True, return_images=False, return_labels=True, return_gt_models=False):
        self.dataset = HopeF(data_dir_path, split, cache_data=cache_data, normalize_coords=normalize_coords,
                             return_images=return_images, return_gt_models=return_gt_models)
        self.max_num_points = max_num_points
        self.permute_points = permute_points
        self.return_images = return_images
        self.return_labels = return_labels

    def denormalise(self, X):
        scale = np.max(self.dataset.img_size) / 2.0
        off = (self.dataset.img_size[1] / 2.0, self.dataset.img_size[0] / 2.0)
        p1 = X[..., 0:2] * scale
        p2 = X[..., 0:2] * scale
        p1[..., 0] += off[0]
        p1[..., 1] += off[1]
        p2[..., 0] += off[0]
        p2[..., 1] += off[1]

        return p1, p2

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        datum = self.dataset[key]

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

        if num_actual_points < max_num_points:
            for i in range(num_actual_points, max_num_points, num_actual_points):
                rest = max_num_points-i
                num = min(rest, num_actual_points)
                points[i:i+num, :] = points[0:num, :].copy()
                labels[i:i+num] = labels[0:num].copy()

        if 'img1' in datum.keys():
            image = datum['img1']
        else:
            image = 0

        imgsize = np.array([(1024, 1024), (1024, 1024)])

        if self.dataset.return_gt_models:
            return points, points, labels, datum["gt"], image, imgsize
        else:
            return points, points, labels, 0, image, imgsize



def make_vis():
    import matplotlib
    import matplotlib.pyplot as plt

    random.seed(0)

    dataset = HopeF("./datasets/hope", "all", cache_data=False, normalize_coords=False, return_images=True)

    target_folder = "./tmp/fig/hopef"

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

        img1g = rgb2gray(img1) * 0.5 + 128
        img2g = rgb2gray(img2) * 0.5 + 128

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

        axs[0].imshow(img1)
        axs[1].imshow(img2)
        fig.tight_layout()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(target_folder, "%02d_%03d_orig.png" % (num_models, idx)), bbox_inches='tight',
                    pad_inches=0)
        plt.close()
