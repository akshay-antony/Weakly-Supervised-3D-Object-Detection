import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def plot_proposals_gt(gt, proposal):
    image = np.zeros((800,700), dtype=np.uint8)
    gt = gt.astype(np.int32)
    proposal = proposal.astype(np.int32)

    # plotting gt boxes
    for i in range(gt.shape[0]):
        x1 = gt[i, 0]
        y1 = gt[i, 1]
        x2 = gt[i, 2]
        y2 = gt[i, 3]
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    # plotting proposal boxes
    for i in range(proposal.shape[0]):
        x1 = proposal[i, 0]
        y1 = proposal[i, 1]
        x2 = proposal[i, 2]
        y2 = proposal[i, 3]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    img = plt.imread(image)
    plt.imshow(img)
    plt.show()