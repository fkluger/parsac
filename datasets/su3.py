import os
import torch.utils.data
import numpy as np
import skimage
from pylsd.lsd import lsd
import pickle
import csv
from glob import glob
import utils.residual_functions

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def augment_sample(datum):
    M = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    if np.random.uniform(0, 1) > 0.5:
        F = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        M = F @ M
    if np.random.uniform(0, 1) > 0.5:
        F = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        M = F @ M
        Mi = np.linalg.inv(M).T

    lines = datum["line_segments"]
    p1 = (M @ lines[:, 0:3].T).T
    p2 = (M @ lines[:, 3:6].T).T
    l = (Mi @ lines[:, 6:9].T).T
    c = (M @ lines[:, 9:12].T).T
    lines[:, 0:3] = p1
    lines[:, 3:6] = p2
    lines[:, 6:9] = l
    lines[:, 9:12] = c

    vps = (M @ datum["VPs"].T).T

    datum["line_segments"] = lines
    datum["VPs"] = vps

    return datum


def remove_outliers(vps, lsd_line_segments, threshold=1-np.cos(0.5*np.pi/180.0)):

    lsd_normed = lsd_line_segments.copy()
    lsd_normed[:, 0:4] -= 256
    lsd_normed[:, 0:4] /= 256.

    line_segments = np.zeros((lsd_line_segments.shape[0], 12))
    for li in range(line_segments.shape[0]):
        p1 = np.array([lsd_normed[li, 0], lsd_normed[li, 1], 1])
        p2 = np.array([lsd_normed[li, 2], lsd_normed[li, 3], 1])
        centroid = 0.5 * (p1 + p2)
        line = np.cross(p1, p2)
        line /= np.linalg.norm(line[0:2])
        line_segments[li, 0:3] = p1
        line_segments[li, 3:6] = p2
        line_segments[li, 6:9] = line
        line_segments[li, 9:12] = centroid

    residuals = utils.residual_functions.vanishing_point(torch.from_numpy(line_segments)[None, ...].cuda(), torch.from_numpy(vps).cuda()).cpu().numpy()

    min_residuals = np.min(residuals, axis=0)

    inliers = min_residuals < threshold

    lsd_line_segments = lsd_line_segments[np.where(inliers)]

    return lsd_line_segments


def label_lines(vps, line_segments, threshold=1-np.cos(2.0*np.pi/180.0)):

    residuals = utils.residual_functions.vanishing_point(torch.from_numpy(line_segments)[None, ...].cuda(), torch.from_numpy(vps).cuda()).cpu().numpy()

    min_residuals = np.min(residuals, axis=0)

    inliers = min_residuals < threshold

    labels = np.argmin(residuals, axis=0) + 1
    labels *= inliers

    return labels


def add_outliers(line_segments, outlier_percentage=0.0):

    N = line_segments.shape[0]

    No = int(N * outlier_percentage/(1-outlier_percentage))

    if outlier_percentage == 0 or No < 1:
        return line_segments

    outlier = np.zeros((No, line_segments.shape[1])).astype(np.float32)
    outlier[:, 0:2] = np.random.uniform(0, 511, (No, 2))
    outlier[:, 2:4] = outlier[:, 0:2] + np.random.uniform(-50, 50, (No, 2))
    outlier = np.clip(outlier, a_min=0, a_max=512)

    lines = np.concatenate([line_segments, outlier], axis=0)

    return lines


def prepare_sample(sample, max_num_lines, max_num_vps, augment=False):

    if augment:
        sample = augment_sample(sample)

    if max_num_lines < 0:
        max_num_lines = sample['line_segments'].shape[0]
    else:
        max_num_lines = max_num_lines

    lines = np.zeros((max_num_lines, 12)).astype(np.float32)
    vps = np.zeros((max_num_vps, 3)).astype(np.float32)

    np.random.shuffle(sample['line_segments'])

    num_actual_line_segments = np.minimum(sample['line_segments'].shape[0], max_num_lines)
    lines[0:num_actual_line_segments, :] = sample['line_segments'][0:num_actual_line_segments, :12].copy()
    if num_actual_line_segments < max_num_lines:
        rest = max_num_lines - num_actual_line_segments
        lines[num_actual_line_segments:num_actual_line_segments + rest, :] = lines[0:rest, :].copy()

    num_actual_vps = np.minimum(sample['VPs'].shape[0], max_num_vps)
    vps[0:num_actual_vps, :] = sample['VPs'][0:num_actual_vps]

    centroids = lines[:, 9:11]
    lengths = np.linalg.norm(lines[:, 0:3] - lines[:, 3:6], axis=-1)[:, None]
    vectors = lines[:, 0:3] - lines[:, 3:6]
    angles = np.abs(np.arctan2(vectors[:, 0], vectors[:, 1]))[:, None]

    features = np.concatenate([centroids, lengths, angles], axis=-1)

    return features, lines, sample['labels'], vps, sample['image'], 0


class SU3(torch.utils.data.Dataset):

    def __init__(self, rootdir, split, max_num_lines=512, normalise_coords=False, augmentation=False,
                 deeplsd_folder=None, cache=True, ablation_outlier_ratio=-1, ablation_noise=0, return_dict=False,
                 generate_labels=False):

        self.rootdir = rootdir
        self.deeplsd_folder = deeplsd_folder
        self.ablation_noise = ablation_noise
        self.ablation_outlier_ratio = ablation_outlier_ratio
        filelist = sorted(glob(f"{rootdir}/*/*.png"))

        self.split = split
        division = int(len(filelist) * 0.1)
        print("num of valid/test", division)
        if split == "train":
            num_train = int(len(filelist) * 0.8 * 1)
            self.filelist = filelist[2 * division: 2 * division + num_train]
            self.size = len(self.filelist)
            print("subset for training: percentage ", 1, num_train)
        elif split == "valid":
            self.filelist = [f for f in filelist[division:division*2] if "a1" not in f]
            self.size = len(self.filelist)
        elif split == "test":
            self.filelist = [f for f in filelist[:division] if "a1" not in f]
            self.size = len(self.filelist)
        elif split == "all":
            self.filelist = [f for f in filelist if "a1" not in f]
            self.size = len(self.filelist)
        print(f"n{split}:", len(self.filelist))

        self.augmentation = augmentation
        self.return_dict = return_dict
        self.max_num_lines = max_num_lines
        self.normalise_coords = normalise_coords
        self.generate_labels = generate_labels

        f = 2.1875 * 256
        c = 256
        self.K = np.array([[f, 0, c], [0, -f, c], [0, 0, 1]])
        self.S = np.array([[1. / c, 0, -1.], [0, 1. / c, -1.], [0, 0, 1]])
        if self.normalise_coords:
            self.SK = self.S @ self.K
        else:
            self.SK = self.K

        self.cache_dir = None
        if cache:
            cache_folders = ["/phys/ssd/tmp/su3_new", "/phys/ssd/slurmstorage/tmp/su3_new", "/tmp/su3_new", "/phys/intern/tmp/su3_new", ]
            for cache_folder in cache_folders:
                try:
                    cache_folder = os.path.join(cache_folder, split)
                    if deeplsd_folder is not None:
                        cache_folder = os.path.join(cache_folder, "deeplsd")
                    os.makedirs(cache_folder, exist_ok=True)
                    self.cache_dir = cache_folder
                    print("%s is cache folder" % cache_folder)
                    break
                except:
                    print("%s unavailable" % cache_folder)

    def denormalise(self, X):
        p1 = X[..., :2] * 256 + 256
        p2 = X[..., 3:5] * 256 + 256

        return p1, p2


    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, k):

        if k >= len(self.filelist):
            raise IndexError

        sample = None

        if self.cache_dir is not None:
            cache_path = os.path.join(self.cache_dir, "%09d.pkl" % k)
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    sample = pickle.load(f)

        if sample is None or not ("image" in sample.keys()):
            iname = self.filelist[k % len(self.filelist)]
            image_rgb = skimage.io.imread(iname).astype(float)[:, :, :3]
            image = rgb2gray(image_rgb)

            with np.load(iname.replace(".png", "_label.npz")) as npz:
                vds = npz["vpts"]
                vps = (self.SK @ vds.T).T

            if self.deeplsd_folder:
                file_seq = iname.split("/")[-2]
                file_idx = iname.split("/")[-1].split(".")[-2]
                target_dir = os.path.join(self.deeplsd_folder, file_seq)
                deeplsd_file = os.path.join(target_dir, file_idx + ".csv")

                lsd_line_segments = []
                with open(deeplsd_file, 'r') as csv_file:
                    reader = csv.reader(csv_file, delimiter=' ')
                    for line in reader:
                        p1x = float(line[0])
                        p1y = float(line[1])
                        p2x = float(line[2])
                        p2y = float(line[3])
                        lsd_line_segments += [np.array([p1x, p1y, p2x, p2y])]
                lsd_line_segments = np.vstack(lsd_line_segments)
            else:
                lsd_line_segments = lsd(image)

            if self.ablation_outlier_ratio >= 0:
                lsd_line_segments = remove_outliers(vps, lsd_line_segments)
                lsd_line_segments = add_outliers(lsd_line_segments, self.ablation_outlier_ratio)

            if self.ablation_noise > 0:
                lsd_line_segments[:, :4] += np.random.normal(0, self.ablation_noise, lsd_line_segments[:, :4].shape)

            if self.normalise_coords:
                lsd_line_segments[:,0:4] -= 256
                lsd_line_segments[:,0:4] /= 256.

            line_segments = np.zeros((lsd_line_segments.shape[0], 12))
            for li in range(line_segments.shape[0]):
                p1 = np.array([lsd_line_segments[li,0], lsd_line_segments[li,1], 1])
                p2 = np.array([lsd_line_segments[li,2], lsd_line_segments[li,3], 1])
                centroid = 0.5*(p1+p2)
                line = np.cross(p1, p2)
                line /= np.linalg.norm(line[0:2])
                line_segments[li, 0:3] = p1
                line_segments[li, 3:6] = p2
                line_segments[li, 6:9] = line
                line_segments[li, 9:12] = centroid

            sample = {"line_segments": line_segments, "VPs": vps, "image": image_rgb, "labels": 0}

            if self.cache_dir is not None:
                cache_path = os.path.join(self.cache_dir, "%09d.pkl" % k)
                if not os.path.exists(cache_path):
                    with open(cache_path, 'wb') as f:
                        # Pickle the 'data' dictionary using the highest protocol available.
                        pickle.dump(sample, f, pickle.HIGHEST_PROTOCOL)


        if self.generate_labels:
            vps = sample["VPs"]
            line_segments = sample["line_segments"]
            labels = label_lines(vps, line_segments)
            sample["labels"] = labels

        if self.return_dict:
            return sample

        data = prepare_sample(sample, self.max_num_lines, 3, augment=self.augmentation)

        return data

def save_mat(split):
    import scipy.io
    dataset = SU3("/data/scene_understanding/SU3", split, return_dict=True)

    target_folder = "/data/kluger/datasets/SU3/matlab/%s" % split
    os.makedirs(target_folder, exist_ok=True)

    filename = os.path.join(target_folder, "../intrinsic_camera.mat")
    scipy.io.savemat(filename, {"K": dataset.K})

    print("num images: ", len(dataset))
    for idx, sample in enumerate(dataset):
        print("%d / %d" % (idx+1, len(dataset)))
        img1 = rgb2gray(sample["image"])[..., None]
        filename = os.path.join(target_folder, "%04d.mat" % idx)
        scipy.io.savemat(filename, {"image": img1, "lines": sample["line_segments"], "VPs": sample["VPs"]})

def make_vis():
    import matplotlib
    import matplotlib.pyplot as plt

    def line_vp_distances(lines, vps):
        distances = np.zeros((lines.shape[0], vps.shape[0]))

        for li in range(lines.shape[0]):
            for vi in range(vps.shape[0]):
                vp = vps[vi, :]
                line = lines[li, 6:9]
                centroid = lines[li, 9:12]
                constrained_line = np.cross(vp, centroid)
                constrained_line /= np.linalg.norm(constrained_line[0:2])

                distance = np.arccos(np.abs((line[0:2] * constrained_line[0:2]).sum(axis=0))) * 180.0 / np.pi

                distances[li, vi] = distance
        return distances

    dataset = SU3("datasets/su3", "test", cache=False, return_dict=True)

    target_folder = "./tmp/fig/su3_examples"

    os.makedirs(target_folder, exist_ok=True)

    for idx in range(len(dataset)):
        vps = dataset[idx]['VPs']
        num_vps = vps.shape[0]

        ls = dataset[idx]['line_segments']
        vp = dataset[idx]['VPs']
        vp[:, 0] /= vp[:, 2]
        vp[:, 1] /= vp[:, 2]
        vp[:, 2] /= vp[:, 2]

        distances = line_vp_distances(ls, vp)
        closest_vps = np.argmin(distances, axis=1) + 1
        closest_vps[np.nonzero(np.min(distances, axis=1) > 5.0)[0]] = 0

        image = dataset[idx]['image'].astype(np.uint8)

        hues = [0, 338, 208, 45, 170, 99, 310, 255, 80, 190, 230, 120]

        fig2 = plt.figure(figsize=(4 * 2, 4 * 2))
        ax2 = fig2.add_subplot()
        ax2.set_aspect('equal', 'box')
        ax2.axis('off')

        if image is not None:
            grey_image = rgb2gray(image) * 0.5 + 128
            ax2.imshow(grey_image, cmap='Greys_r', vmin=0, vmax=255)

        for li in range(ls.shape[0]):
            vpidx = closest_vps[li]
            if vpidx:
                hue = hues[vpidx % len(hues)] / 360.0
                sat = 1
                val = 1 if vpidx else 0
                c = matplotlib.colors.hsv_to_rgb([hue, sat, val])

                ax2.plot([ls[li, 0], ls[li, 3]], [ls[li, 1], ls[li, 4]], c=c, lw=3)

        fig2.tight_layout()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(os.path.join(target_folder, "%03d_vp_%d.png" % (idx, num_vps)), bbox_inches='tight',
                    pad_inches=0)
        plt.imsave(os.path.join(target_folder, "%03d_orig.png" % idx), image)
        plt.close()
