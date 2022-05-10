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