import torch.utils.data
import numpy as np
from datasets.nyu_vp import nyu
from datasets.yud_plus import yud
import utils.residual_functions

def label_lines(vps, line_segments, threshold=1-np.cos(2.0*np.pi/180.0)):

    residuals = utils.residual_functions.vanishing_point(torch.from_numpy(line_segments)[None, ...].cuda(), torch.from_numpy(vps).cuda()).cpu().numpy()

    min_residuals = np.min(residuals, axis=0)

    inliers = min_residuals < threshold

    labels = np.argmin(residuals, axis=0) + 1
    labels *= inliers

    return labels

def prepare_sample(sample, max_num_lines, max_num_vps, generate_labels=False):
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

    if generate_labels:
        labels = label_lines(vps, lines)
    else:
        labels = 0

    return features, lines, labels, vps, 0, 0


class NYUVP(torch.utils.data.Dataset):

    def __init__(self, split, max_num_lines=512, max_num_vps=8, use_yud=False, use_yud_plus=False, deeplsd_folder=None,
                 cache=True, generate_labels=False):
        if use_yud:
            self.dataset = yud.YUDVP(split=split, normalize_coords=True, data_dir_path="./datasets/yud_plus/data",
                                     yudplus=use_yud_plus, keep_in_memory=cache, external_lines_folder=deeplsd_folder)
        else:
            self.dataset = nyu.NYUVP(split=split, normalise_coordinates=True, data_dir_path="./datasets/nyu_vp/data",
                                     keep_data_in_memory=cache, external_lines_folder=deeplsd_folder)
        self.max_num_lines = max_num_lines
        self.max_num_vps = max_num_vps
        self.generate_labels = generate_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, k):
        sample = self.dataset[k]

        return prepare_sample(sample, self.max_num_lines, self.max_num_vps, self.generate_labels)
