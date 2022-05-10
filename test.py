from visualize_dataset_new import plot_proposals_gt
from dataset import KITTIBEV
import torch
from torch.utils.data import DataLoader, Dataset


valid_data_list_filename = "./valid_data_list_after_threshold.txt"
lidar_folder_name = "./data"

dataset = KITTIBEV(valid_data_list_filename=valid_data_list_filename, 
                   lidar_folder_name=lidar_folder_name)

import numpy as np
dataloader = DataLoader(dataset, batch_size=1)
i = 0
for data in dataloader:
    # if i <= 10:
    #     i += 1 
    #     continue
    bev = data['bev']
    #labels = data['labels']
    gt_boxes = data['gt_boxes'].squeeze(0)
    proposals = data['proposals'].squeeze(0)
    #proposals = torch.cuda.FloatTensor(proposals)
    #gt_class_list = data['gt_class_list'].cuda()
    gt_boxes = gt_boxes.cpu().detach().numpy()
    proposals = proposals.cpu().detach().numpy()
    # proposals[:, 0] = 800 - proposals[:, 0]
    # proposals[:, 2] = 800 - proposals[:, 2]
    # print(gt_boxes.shape, proposals.shape)
    # plot_proposals_gt(gt_boxes.astype(np.int32), proposals.astype(np.int32), bev[0])
    i += 1
    break
    if i >= 20:
        break