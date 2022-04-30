
from multiprocessing import reduction
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from WSDNN_Resnet import  WSDNN_Resnet
from dataset import KITTICam
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.ops import nms
from post_processing import calculate_ap
import wandb
import math
import sklearn
import sklearn.metrics
from visualize_dataset_new import plot_bev
from loss import FocalLoss
import torch.nn.functional as F

def train(train_loader, 
          model, 
          loss_fn,
          optimizer, 
          test_loader,
          num_classes=2):
    loss_bce_total = 0.0
    loss_total = 0.0
    data_count = 0.0
    total_target = torch.zeros((0, num_classes)).cuda()
    total_preds = torch.zeros((0, num_classes)).cuda()
    for iter, data in tqdm(enumerate(train_loader),
                           total=len(train_loader),
                           leave=False):
        model = model.train()
        bev = data['image'].cuda()
        labels = data['labels'].cuda()
        #gt_boxes = data['gt_boxes'].cuda()
        proposals = data['proposals'].squeeze().float().cuda()
        proposals = torch.cuda.FloatTensor(proposals)
        #gt_class_list = data['gt_class_list'].cuda()
        #with torch.cuda.amp.autocast():
        preds = model(bev, proposals)
        preds_class = preds.sum(dim=0).reshape(1, -1)
        # preds_class_sigmoid = torch.sigmoid(preds_class)
        # total_preds = torch.cat([total_preds, preds_class_sigmoid], dim=0)
        # total_target = torch.cat([total_target, labels], dim=0)
        preds_class = torch.clamp(preds_class, 0, 1)
        loss = loss_fn(preds_class, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        #loss_bce_total += F.binary_cross_entropy_with_logits(preds_class, labels, reduction='sum').item()
        loss_total += loss.item() * bev.shape[0]
        data_count += bev.shape[0]
        if iter%500 == 0 and iter != 0:
            #map_class = map_classification(total_preds, total_target)
            wandb.log({"Loss":loss_total / data_count})
            print("Focal Loss: ", loss_total / data_count) #, " BCE loss: ", loss_bce_total / data_count) #,  " mAP: ", map_class)
        # if iter%5000 == 0 and iter != 0:
        #     model.eval()
        #     validate(test_loader, model, loss_fn)
    return loss_total / data_count

def validate(test_loader, 
             model, 
             loss_fn, 
             score_threshold=0.005,
             nms_iou_threshold=0.5,
             iou_list = [0.05, 0.1, 0.2, 0.3, 0.4],
             inv_class=None,
             direct_class=None):
    np.random.seed(2)
    num_classes = 1
    loss_total = 0.0
    data_count = 0.0
    all_gt_boxes = torch.zeros((0, 6))
    all_pred_boxes = torch.zeros((0, 7))
    plotting_idxs = np.random.randint(0, 500, (50))

    with torch.no_grad():
        for iter, data in tqdm(enumerate(test_loader),
                            total=len(test_loader),
                            leave=False):
            plotting_proposals = torch.zeros((0, 5))
            plotting_gts = torch.zeros((0, 5))
            bev = data['image'].cuda()
            labels = data['labels'].cuda()
            gt_boxes = data['gt_boxes'].reshape(-1, 4) #.cuda()
            proposals = data['proposals'].squeeze().float().cuda()
            gt_class_list = data['gt_class_list'].reshape(-1) #.cuda()

            cls_probs = model(bev, proposals)
            preds_class = cls_probs.sum(dim=0).reshape(1, -1)
            loss = loss_fn(preds_class, labels)
            loss_total += loss.item()
            data_count += bev.shape[0]

            for i in range(gt_boxes.shape[0]):
                modified_boxes = torch.cat([torch.tensor([iter, gt_class_list[i]]), gt_boxes[i]]).reshape(1, -1)
                all_gt_boxes = torch.cat([all_gt_boxes, modified_boxes], dim=0)
                plotting_gts = torch.cat([plotting_gts,
                                          modified_boxes[0, 1:].reshape(1, -1)], dim=0)

            for class_num in range(num_classes):
                curr_class_scores = cls_probs[:, class_num]
                valid_score_idx = torch.where(curr_class_scores >= score_threshold)
                valid_scores = curr_class_scores[valid_score_idx]
                valid_proposals = proposals[valid_score_idx]
                retained_idx = nms(valid_proposals, valid_scores, nms_iou_threshold)
                retained_scores = valid_scores[retained_idx]
                retained_proposals = valid_proposals[retained_idx]

                class_num_for_plotting = torch.ones((retained_proposals.shape[0], 1)) * class_num
                plotting_proposals = torch.cat([plotting_proposals,
                                                torch.cat([retained_proposals.detach().cpu(), 
                                                           class_num_for_plotting], dim=1)], dim=0)

                for i in range(retained_proposals.shape[0]):
                    modified_pred_boxes = torch.cat([torch.tensor([iter, class_num, retained_scores[i]]), 
                                                                retained_proposals[i].detach().cpu()]).reshape(1, -1)
                    all_pred_boxes = torch.cat([all_pred_boxes, modified_pred_boxes], dim=0)

            if iter in plotting_idxs:
                all_boxes = []
                all_gt_plotting_boxes = []
                raw_image = plot_bev(bev[0].detach().cpu())

                for idx in range(plotting_proposals.shape[0]):
                    box_data = {"position": {
                        "minX": plotting_proposals[idx, 0].item() / 1224,
                        "minY": plotting_proposals[idx, 1].item() / 370,
                        "maxX": plotting_proposals[idx, 2].item() / 1224,
                        "maxY": plotting_proposals[idx, 3].item() / 370},
                        "class_id": int(plotting_proposals[idx, 4].item()),
                        "box_caption": inv_class[int(plotting_proposals[idx][4])],
                        }
                    all_boxes.append(box_data)
                

                for idx in range(plotting_gts.shape[0]):
                    box_data_new = {"position": {
                        "minX": plotting_gts[idx, 1].item() / 1224,
                        "minY": plotting_gts[idx, 2].item() / 370,
                        "maxX": plotting_gts[idx, 3].item() / 1124,
                        "maxY": plotting_gts[idx, 4].item() / 370},
                        "class_id": int(plotting_gts[idx, 0].item()),
                        "box_caption": inv_class[int(plotting_gts[idx][0])],
                        }
                    all_gt_plotting_boxes.append(box_data_new)
                    
                box_image = wandb.Image(raw_image, 
                                        boxes={"predictions":
                                        {"box_data": all_boxes,
                                        "class_labels": inv_class},
                                             "ground_truth":
                                        {"box_data": all_gt_plotting_boxes,
                                        "class_labels": inv_class}
                                        })
                wandb.log({"Image proposals " + str(iter): box_image})
                box_image = wandb.Image(raw_image, 
                                        boxes= {"predictions":
                                        {"box_data": all_gt_plotting_boxes,
                                        "class_labels": inv_class}
                                        })
                wandb.log({"Image gt " + str(iter): box_image})
                
    for iou in iou_list:
        #print(all_gt_boxes.shape, all_gt_boxes.shape)
        AP = calculate_ap(all_pred_boxes, all_gt_boxes, iou, inv_class=inv_class, total_cls_num=num_classes)
        mAP = 0 if len(AP) == 0 else sum(AP) / len(AP)
        #return mAP.item(), AP
        wandb.log({"map@ " + str(iou): mAP})
        print("Iou ", iou, " mAP ", mAP)
    return mAP


if __name__ == '__main__':
    valid_data_list_filename = "./valid_data_list_after_threshold.txt"
    lidar_folder_name = "/media/akshay/Data/KITTI/"
    dataset = KITTICam(valid_data_list_filename=valid_data_list_filename, 
                            lidar_folder_name=lidar_folder_name)
    wandb.init("WSDNNResnet")
    epochs = 10
    model = WSDNN_Resnet()

    for params in model.encoder.parameters():
        params.requires_grad = False

    train_dataset_length = int(0.70 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_dataset_length,
                                                        len(dataset) - train_dataset_length],
                                                        generator=torch.Generator().manual_seed(10))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(len(train_dataset), len(test_dataset))

    #scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    #loss_fn = FocalLoss(alpha=0.25, gamma=2)
    model = model.cuda()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)
    for i in range(epochs):
        # if i%1 == 0:
        #     model = model.eval()
        #     mAP = validate(test_loader, 
        #                   model, 
        #                   loss_fn, 
        #                   inv_class=dataset.inv_class, 
        #                   direct_class=dataset.class_to_int)
        model = model.train()
        loss = train(train_loader, model, loss_fn, optimizer, test_loader)
        print("Epoch average Loss: ", loss)
        torch.save(model.state_dict(), "model_res.pth")
        torch.save(optimizer.state_dict(), "opt_res.pth")
        if i%1 == 0:
            model = model.eval()
            mAP = validate(test_loader, 
                          model, 
                          loss_fn, 
                          inv_class=dataset.inv_class, 
                          direct_class=dataset.class_to_int)