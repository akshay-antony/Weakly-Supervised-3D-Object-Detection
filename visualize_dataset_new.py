import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from tqdm import tqdm

def lidar_preprocess(self, scan):
    velo = scan
    velo_processed = np.zeros(self.geometry['input_shape'], dtype=np.float32)
    intensity_map_count = np.zeros((velo_processed.shape[0], velo_processed.shape[1]))
    for i in range(velo.shape[0]):
        if self.point_in_roi(velo[i, :]):
            x = int((velo[i, 1]-self.geometry['L1']) / 0.1)
            y = int((velo[i, 0]-self.geometry['W1']) / 0.1)
            z = int((velo[i, 2]-self.geometry['H1']) / 0.1)
            velo_processed[x, y, z] = 1
            velo_processed[x, y, -1] += velo[i, 3]
            intensity_map_count[x, y] += 1
    velo_processed[:, :, -1] = np.divide(velo_processed[:, :, -1],  intensity_map_count, \
                    where=intensity_map_count!=0)
    return velo_processed

def create_corners(x_min, y_min, x_max, y_max):
    corners = np.array([[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]])
    return corners

def plot_proposals_gt(gt, proposal, pc_feature):
    gt = gt.astype(np.int32)
    proposal = proposal.astype(np.int32)
    pc_feature = pc_feature.permute(1,2,0)

    pc_feature = pc_feature.numpy()
    pc_feature = pc_feature[::-1, :, :-1]

    val = 1 - pc_feature.max(axis=2)
    val = val.astype(np.uint8)

    intensity = np.zeros((pc_feature.shape[0], pc_feature.shape[1], 3))
    intensity[:, :, 0] = val
    intensity[:, :, 1] = val
    intensity[:, :, 2] = val
    intensity = intensity.astype(np.uint8)*255

    for box in gt:
        plot_corners = create_corners(box[0], box[1], box[2], box[3])
        plot_corners[:, 0] = intensity.shape[0] - plot_corners[:, 0]
        plot_corners = plot_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(intensity, [plot_corners], True, (0, 0, 255), 2)
    
    for box in proposal:
        plot_corners = create_corners(box[0], box[1], box[2], box[3])
        plot_corners[:, 0] = intensity.shape[0] - plot_corners[:, 0]
        plot_corners = plot_corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(intensity, [plot_corners], True, (0, 255, 0), 2)

    plt.imshow(intensity)
    plt.show()
    

    # fig, ax = plt.subplots()
    # ax.imshow(image)

    # # plotting gt boxes
    # for i in range(gt.shape[0]):
    #     y1 = gt[i, 0]
    #     x1 = gt[i, 1]
    #     y2 = gt[i, 2]
    #     x2 = gt[i, 3]
    #     height = y2 - y1
    #     width = x2 - x1
    #     corner = (x1, y1)
    #     rect = patches.Rectangle(corner, width, height, linewidth=1, edgecolor='r', facecolor='none')
    #     ax.add_patch(rect)

    # # plotting proposal boxes
    # for i in range(proposal.shape[0]):
    #     y1 = proposal[i, 0]
    #     x1 = proposal[i, 1]
    #     y2 = proposal[i, 2]
    #     x2 = proposal[i, 3]
    #     height = y2 - y1
    #     width = x2 - x1
    #     corner = (x1, y1)
    #     rect = patches.Rectangle(corner, width, height, linewidth=1, edgecolor='b', facecolor='none')
    #     ax.add_patch(rect)
    
    plt.show()
    


if __name__ == "__main__":
    gt = np.array([[0, 0, 100, 100], [100, 100, 200, 200], [200, 200, 300, 300]])
    proposal = np.array([[0, 0, 50, 50], [100, 100, 100, 100], [200, 200, 400, 400]])
    plot_proposals_gt(gt, proposal)