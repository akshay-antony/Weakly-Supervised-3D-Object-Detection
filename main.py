from multiprocessing import reduction
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from WsdnnPIXOR import WSDDNPIXOR
from dataset import KITTIBEV
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.ops import nms
from post_processing import calculate_ap
import wandb


def load_pretrained(model, filename='40epoch'):
    own_state = model.state_dict()
    state_dict = torch.load(filename)
    for name, param in state_dict.items():
        if name not in own_state:
                continue
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

def train(train_loader, model, loss_fn, optmizier):
    loss_total = 0.0
    data_count = 0.0
    for iter, data in tqdm(enumerate(train_loader),
                           total=len(train_loader),
                           leave=False):
        bev = data['bev'].cuda()
        labels = data['labels'].cuda()
        #gt_boxes = data['gt_boxes'].cuda()
        proposals = data['proposals'].squeeze().cuda()
        #gt_class_list = data['gt_class_list'].cuda()
        preds = model(bev, proposals)
        preds_class = preds.sum(dim=0)
        loss = loss_fn(preds_class, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()
        data_count += bev.shape[0]
        if iter%1000 == 0:
            wandb.log({"Loss": loss.item()})
            print("Loss: ", loss.item())
    return loss_total / data_count

def validate(test_loader, 
             model, 
             loss_fn, 
             score_threshold=0.05,
             nms_iou_threshold=0.3,
             iou_list = [0.05, 0.1, 0.2, 0.3, 0.4]):
    num_classes = 9
    loss_total = 0.0
    data_count = 0.0
    all_gt_boxes = torch.zeros((0, 6))
    all_pred_boxes = torch.zeros((0, 7))
    for iter, data in tqdm(enumerate(test_loader),
                           total=len(test_loader),
                           leave=False):
        bev = data['bev'].cuda()
        labels = data['labels'].cuda()
        gt_boxes = data['gt_boxes'].reshape(-1, 4).cuda()
        proposals = data['proposals'].squeeze().cuda()
        gt_class_list = data['gt_class_list'].reshape(-1).cuda()

        with torch.no_grad():
            cls_probs = model(bev, proposals)
        preds_class = cls_probs.sum(dim=0)
        loss = loss_fn(preds_class, labels)
        loss_total += loss.item()
        data_count += bev.shape[0]

        for i in range(gt_boxes.shape[0]):
            modified_boxes = torch.cat([torch.tensor([iter, gt_class_list[i]]), gt_boxes[i]]).reshape(1, -1)
            all_gt_boxes = torch.cat([all_gt_boxes, modified_boxes], dim=0)

        for class_num in range(num_classes):
            curr_class_scores = cls_probs[:, class_num]
            valid_score_idx = torch.where(curr_class_scores >= score_threshold)
            valid_scores = curr_class_scores[valid_score_idx]
            valid_proposals = proposals[valid_score_idx]
            retained_idx = nms(valid_proposals, valid_scores, nms_iou_threshold)
            retained_scores = valid_scores[retained_idx]
            retained_proposals = valid_proposals[retained_idx]

            for i in range(retained_proposals.shape[0]):
                modified_pred_boxes = torch.cat([torch.tensor([iter, class_num, retained_scores[i]]), 
                                                               retained_proposals[i]]).reshape(1, -1)
                all_pred_boxes = torch.cat([all_pred_boxes, modified_pred_boxes], dim=0)
    
    for iou in iou_list:
        AP = calculate_ap(all_pred_boxes, all_gt_boxes, iou)
        mAP = 0 if len(AP) == 0 else sum(AP) / len(AP)
        #return mAP.item(), AP
        wandb.log({"map@ " + str(iou): mAP.item()})
        print("Iou ", iou, " mAP ", mAP.item())
    return mAP

if __name__ == '__main__':
    wandb.init("WSDNNPIXOR")
    epochs = 10
    model = WSDDNPIXOR()
    load_pretrained(model)
    valid_data_list_filename = "valid_filenames.txt"
    lidar_folder_name = "/media/akshay/Data/KITTI/"

    dataset = KITTIBEV(valid_data_list_filename=valid_data_list_filename, 
                       lidar_folder_name=lidar_folder_name)

    train_dataset_length = int(0.7 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_dataset_length,
                                                        len(dataset) - train_dataset_length],
                                                        generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(len(train_dataset), len(test_dataset))

    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(epochs):
        model = model.train()
        loss = train(train_loader, model, loss_fn, optimizer)
        print("Epoch average Loss: ", loss)

        if i%3 == 0:
            model = model.eval()
            mAP = validate(test_loader, model, loss_fn) 