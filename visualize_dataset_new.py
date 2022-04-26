from operator import is_
import numpy as np
import math
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from tqdm import tqdm
from dataset import KITTIBEV

# def point_in_roi(self, point):
#     if (point[0] - self.geometry['W1']) < 0.01 or (self.geometry['W2'] - point[0]) < 0.01:
#         return False
#     if (point[1] - self.geometry['L1']) < 0.01 or (self.geometry['L2'] - point[1]) < 0.01:
#         return False
#     if (point[2] - self.geometry['H1']) < 0.01 or (self.geometry['H2'] - point[2]) < 0.01:
#         return False
#     return True

# def lidar_preprocess(self, scan):
#     velo = scan
#     velo_processed = np.zeros(self.geometry['input_shape'], dtype=np.float32)
#     intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
#     for i in range(velo.shape[0]):
#         if self.point_in_roi(velo[i, :]):
#             x = int((velo[i, 1]-self.geometry['L1']) / 0.1)
#             y = int((velo[i, 0]-self.geometry['W1']) / 0.1)
#             z = int((velo[i, 2]-self.geometry['H1']) / 0.1)
#             velo_processed[x, y, z] = 1
#             velo_processed[x, y, -1] += velo[i, 3]
#             intensity_map_count[x, y] += 1
#     velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1],  intensity_map_count, \
#                     where=intensity_map_count!=0)
#     return velo_processed

def create_corners(x_min, y_min, x_max, y_max):
    corners = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
    return corners

def plot_proposals_gt(gt, proposal, pc_feature):
    print("gt: ", gt.shape)
    print("proposal: ", proposal.shape)
    print("pc_feature: ", pc_feature.shape)
    gt = gt.astype(np.int32)
    proposal = proposal.astype(np.int32)
    pc_feature = pc_feature.permute(1,2,0)

    pc_feature = pc_feature.numpy()
    pc_feature = pc_feature[:, :, :-1]

    val = 1 - pc_feature.max(axis=2)
    val = val.astype(np.uint8)

    intensity = np.zeros((pc_feature.shape[0], pc_feature.shape[1], 3))
    intensity[:, :, 0] = val
    intensity[:, :, 1] = val
    intensity[:, :, 2] = val
    intensity = intensity.astype(np.uint8)*255

    fig, ax = plt.subplots()
    ax.imshow(intensity)

    # plotting gt boxes
    for i in range(gt.shape[0]):
        y1 = gt[i, 0]
        x1 = gt[i, 1]
        y2 = gt[i, 2]
        x2 = gt[i, 3]
        height = y2 - y1
        width = x2 - x1
        corner = (x1, y1)
        rect = patches.Rectangle(corner, width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # plotting proposal boxes
    for i in range(proposal.shape[0]):
        y1 = proposal[i, 0]
        x1 = proposal[i, 1]
        y2 = proposal[i, 2]
        x2 = proposal[i, 3]
        height = y2 - y1
        width = x2 - x1
        corner = (x1, y1)
        rect = patches.Rectangle(corner, width, height, linewidth=1, edgecolor='b', facecolor='none')
        ax.add_patch(rect)
    
    #ax.show()
    plt.savefig("temp.png")
    output = Image.open("temp.png")

    plt.show()
    return output

def plot_bev(pc_feature):
    # print("gt: ", gt.shape)
    # print("proposal: ", proposal.shape)
    # print("pc_feature: ", pc_feature.shape)
    # gt = gt.astype(np.int32)
    # proposal = proposal.astype(np.int32)
    pc_feature = pc_feature.permute(1,2,0)

    pc_feature = pc_feature.numpy()
    pc_feature = pc_feature[:, :, :-1]

    val = 1 - pc_feature.max(axis=2)
    val = val.astype(np.uint8)

    intensity = np.zeros((pc_feature.shape[0], pc_feature.shape[1], 3))
    intensity[:, :, 0] = val
    intensity[:, :, 1] = val
    intensity[:, :, 2] = val
    intensity = intensity.astype(np.uint8)*255
    raw_image = Image.fromarray(intensity)
    return raw_image

    


if __name__ == "__main__":
    dataset = KITTIBEV(is_train=True, lidar_folder_name="KITTI", label_folder_name=None, valid_data_list_filename="BAnet/vlr_project/valid_filenames.txt")
    idx = 10
    # for idx in range(10):
    pc_feature = dataset[idx]['bev']
    gt = dataset[idx]['gt_boxes'].numpy()
    print("gt_bbox: ", gt)
    proposal = dataset[idx]['proposals'].squeeze(0).numpy()
    plot_proposals_gt(gt, proposal, pc_feature)