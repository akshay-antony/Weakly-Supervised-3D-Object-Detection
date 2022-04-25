import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    basefilename = "/media/akshay/Data/KITTI/training/bbox/"
    i = 0
    for filename in os.listdir(basefilename):
        if i == 0:
            i += 1
            continue
        boxes = np.empty((0, 6))
        print(filename)
        with open(basefilename + filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                A = line.split(',')
                B = np.array([float(a) for a in A])
                boxes = np.vstack([boxes, B.reshape(1, -1)])
        #break
        #constraint_boxes = (boxes[:, 3] <= 70 and boxes[:, 1] >= -40 and boxes[:, 4] <= 40)
        const_boxes = np.where(boxes[:, 3] <= 70)
        n_boxes = boxes[const_boxes]
        const_boxes = np.where(n_boxes[:, 1] >= -40)
        n_boxes = n_boxes[const_boxes]
        const_boxes = np.where(n_boxes[:, 4] <= 40)
        boxes = n_boxes[const_boxes]
        # boxes[:, 1] >= -40, boxes[:, 4] <= 40)])
        # and #np.any(boxes[:, 4] <= 40)])
        #const_boxes  = boxes
        #boxes = boxes[:, const_boxes]
        #print(const_boxes, boxes.shape)
        area = (boxes[:, 5] - boxes[:, 2]) * (boxes[:, 4] - boxes[:, 1]) * (boxes[:, 3] - boxes[:, 0])
        #area = np.sort(area)
        area_idx = np.where(area >= 0.0)
        boxes_new = boxes[area_idx]
        boxes_2d = np.concatenate([boxes_new[:, 0].reshape(-1, 1), 
                                   boxes_new[:, 1].reshape(-1, 1),
                                   boxes_new[:, 3].reshape(-1, 1), 
                                   boxes_new[:, 4].reshape(-1, 1)], axis=1)
        print(boxes_2d.shape)
        #print(i, boxes.shape, area.max(), area.min(), boxes_new.shape)
        i += 1
        np.save("boxes", boxes_2d)
        break