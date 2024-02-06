'''
Codes for extracting and saving data(points, labels, uv coordinates) only in the view-frustum of RGB
'''
import os
import glob
import pickle

import numpy as np
from PIL import Image


# colors used in  the paper
CUBOID_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (1, 1, 0.5)*255,
    2: (0.5, 1, 1)*255,
    3: (0.5, 0.5, 1)*255,
    4: (0.5, 1, 0.5)*255,
    5: (1, 0.5, 0.5)*255,
    6: (0, 0.5, 0.5)*255,
    7: (0.5, 0.5, 0)*255,
    8: (0.75, 0, 0.75)*255,
    9: (1, 0.75, 0)*255,
    10: (0.5, 0.75, 0.5)*255,
    11: (0.5, 0.75, 1)*255,
    12: (1, 0.5, 0.25)*255,
    13: (1, 1, 0.75)*255,
    14: (0.5, 0, 1)*255,
    15: (0.75, 1, 0.25)*255,
    16: (1, 0.75, 1)*255,
    17: (0.75, 0.5, 0)*255,
    18: (0, 0, 0),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (1, 0.5, 1)*255,
    28: (158., 218., 229.),
}

def save_obj(out, sample, color=None):
    with open(out, 'w') as file:
        if color is None:
            for v1 in sample:
                file.write('v %.4f %.4f %.4f\n' % (v1[0], v1[1], v1[2]))
        else:
            for (v1, c) in zip(sample, color):
                file.write(
                    'v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v1[0], v1[1], v1[2], c[0], c[1], c[2]))  # 0.5*c,0.5*c,0.5*


def save_obj_color_coding(out, samples, labels):
    with open(out, 'w') as file:
        for (v, l) in zip(samples, labels):
            c = CUBOID_COLOR_MAP[l]
            file.write(
                'v %.4f %.4f %.4f %.4f %.4f %.4f\n' % (v[0], v[1], v[2], c[0], c[1], c[2]))

# root = '/home/poscoict/Datasets/SemanticKITTI/dataset/sequences/'
root = 'C:\\posco\\lidar-seg\\dataset\\KITTI-sample\\sequences'
save_root = 'C:\\posco\\lidar-seg\\dataset\\KITTI-sample-sub\\sequences'

seqs = [os.path.join(root, x) for x in os.listdir(root)]

# load pcd and label
for seq in seqs:
    seq_root = os.path.join(root, seq)
    ids = [os.path.basename(x).split('.')[0] for x in glob.glob(seq_root + '/velodyne/*.bin')]
    for id in ids:
        pts = np.fromfile(os.path.join(seq_root, 'velodyne', id + '.bin'), dtype=np.float32).reshape((-1,4))[:,:3] # [n,3]
        label = np.zeros(len(pts), dtype=int)  # [n,1]
        pred = np.ones(len(pts), dtype=int) # [n,1]
        save_obj_color_coding('label.obj', pts, label%28)
        save_obj_color_coding('pred.obj', pts, pred%28)

