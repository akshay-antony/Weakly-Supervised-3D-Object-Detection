import imp
import numpy as np 
import torch 
from tqdm import tqdm
import wandb
import torchvision

def iou(box1, box2):
    """
    Calculates Intersection over Union for two bounding boxes (xmin, ymin, xmax, ymax)
    returns IoU vallue
    """
    if box1[0] > box2[2] or box1[2] < box2[0] or box1[1] > box2[3] or box1[3] < box2[1]:
        return 0 
    x1_common = max(box1[0], box2[0])
    x2_common = min(box1[2], box2[2])
    y1_common = max(box1[1], box2[1])
    y2_common = min(box1[3], box2[3])

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    common_area = (x2_common - x1_common) * (y2_common - y1_common)
    iou = common_area / (area1 + area2 - common_area)
    return iou


def calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.5, inv_class=None, total_cls_num=1):
    AP = []
    class_wise_correct_found = np.zeros((8))
    for class_num in range(total_cls_num):
        # valid_gt_boxes = torch.zeros((0, 6))
        # valid_pred_boxes = torch.zeros((0, 7))

        valid_gt_boxes_ind = torch.where(gt_boxes[:,1] == class_num)
        valid_gt_boxes = gt_boxes[valid_gt_boxes_ind]
        
        valid_pred_boxes_ind = torch.where(pred_boxes[:, 1] == class_num)
        valid_pred_boxes = pred_boxes[valid_pred_boxes_ind]
        
        pred_ind = torch.argsort(valid_pred_boxes[:,2], descending=True)
        valid_pred_boxes = valid_pred_boxes[pred_ind]

        FP = torch.zeros((valid_pred_boxes.shape[0]))
        TP = torch.zeros((valid_pred_boxes.shape[0]))
        total_gts = valid_gt_boxes.shape[0]
        if total_gts == 0:
            print("WHy")
            AP.append(torch.tensor([0]))
            continue
        
        taken_gt_boxes = set()
        for i in range(valid_pred_boxes.shape[0]):
            curr_valid_gt_boxes_ind = torch.where(valid_gt_boxes[:,0] == valid_pred_boxes[i,0])
            curr_valid_gt_boxes = valid_gt_boxes[curr_valid_gt_boxes_ind] 

            best_iou = 0
            for j in range(curr_valid_gt_boxes.shape[0]):
                curr_iou = iou(curr_valid_gt_boxes[j, 2:].reshape(-1), valid_pred_boxes[i, 3:].reshape(-1))
                # curr_iou = torchvision.ops.box_iou(curr_valid_gt_boxes[j, 2:].reshape(-1, 4), 
                #                                    valid_pred_boxes[i, 3:].reshape(-1, 4))
                if curr_iou > best_iou:
                    best_iou = curr_iou
                    best_gt_idx = j 
            
            if best_iou >= iou_threshold:
                if (best_gt_idx, valid_pred_boxes[i,0]) in taken_gt_boxes:
                    FP[i] = 1 
                else:
                    class_wise_correct_found[class_num] += 1
                    taken_gt_boxes.add((best_gt_idx, valid_pred_boxes[i,0]))
                    TP[i] = 1 
            else:
                FP[i] = 1 

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / total_gts
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))

        ap = 0
        for i in range(precisions.shape[0]):
            precisions[i] = torch.tensor([max(precisions[i].item(), torch.max(precisions[i:]).item())])
            if i >= 1:
                ap += precisions[i] * (recalls[i] - recalls[i-1]) 


        AP.append(ap)
        wandb.log({"Class Ap of " + inv_class[class_num] + 
                   " @iou " + str(iou_threshold): ap})
        wandb.log({"Percentage detected of " + 
                    inv_class[class_num] + " @iou " + 
                    str(iou_threshold): class_wise_correct_found[class_num] / total_gts})
        wandb.log({"Found these many correct for " + 
                    inv_class[class_num] + " @iou " + 
                    str(iou_threshold): class_wise_correct_found[class_num]})
        wandb.log({"Found these many total for " + 
                    inv_class[class_num] + " @iou " + 
                    str(iou_threshold): total_gts})
        #AP.append(torch.trapz(precisions, recalls))
    return AP
