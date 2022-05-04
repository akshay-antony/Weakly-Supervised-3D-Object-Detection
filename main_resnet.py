
from multiprocessing import reduction
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from WSDNN_Resnet import  WSDNN_Resnet, WSDNN_Alexnet
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
from torchvision import transforms


def tensor_to_PIL(image):
    """
    converts a tensor normalized image (imagenet mean & std) into a PIL RGB image
    will not work with batches (if batch size is 1, squeeze before using this)
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
        std=[1/0.229, 1/0.224, 1/0.255],
    )

    inv_tensor = inv_normalize(image)
    inv_tensor = torch.clamp(inv_tensor, 0, 1)
    original_image = transforms.ToPILImage()(inv_tensor).convert("RGB")

    return original_image

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
        labels = data['labels'].float().cuda()
        #gt_boxes = data['gt_boxes'].cuda()
        proposals = data['proposals'].squeeze().float().cuda()
        proposals = torch.cuda.FloatTensor(proposals)
        #gt_class_list = data['gt_class_list'].cuda()
        #with torch.cuda.amp.autocast():
        preds = model(bev, proposals)
        preds_class = preds.sum(dim=0).reshape(1, -1)
        #print(preds_class, labels)
        # preds_class_sigmoid = torch.sigmoid(preds_class)
        # total_preds = torch.cat([total_preds, preds_class_sigmoid], dim=0)
        # total_target = torch.cat([total_target, labels], dim=0)
        preds_class = torch.clamp(preds_class, 0, 1)
        #print(labels.shape, preds_class.shape, type(labels), type(preds_class))
        loss = loss_fn(preds_class, labels)
        
        bce_funtion = torch.nn.BCELoss(reduction='sum')
        #loss_bce_total += bce_funtion(preds_class, labels).item()
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
            print("Loss: ", loss_total / data_count , " BCE loss: ", loss_bce_total / data_count) #,  " mAP: ", map_class)
            # loss_total = 0
            # data_count = 0
            # loss_bce_total = 0
        # if iter%5000 == 0 and iter != 0:
        #     model.eval()
        #     validate(test_loader, model, loss_fn)
    return loss_bce_total / data_count

def validate(test_loader, 
             model, 
             loss_fn, 
             score_threshold=0.5,
             nms_iou_threshold=0.5,
             iou_list = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
             inv_class=None,
             direct_class=None,
             dataset=None):
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
            labels = data['labels'].float().cuda()
            gt_boxes = data['gt_boxes'].reshape(-1, 4) #.cuda()
            proposals = data['proposals'].squeeze().float().cuda()
            gt_class_list = data['gt_class_list'].reshape(-1) #.cuda()

            cls_probs = model(bev, proposals)
            #preds_class = cls_probs.sum(dim=0).reshape(1, -1)
            # loss = loss_fn(preds_class, labels)
            # loss_total += loss.item()
            data_count += bev.shape[0]

            for i in range(gt_boxes.shape[0]):
                modified_boxes = torch.cat([torch.tensor([iter, gt_class_list[i]]), gt_boxes[i]]).reshape(1, -1)
                all_gt_boxes = torch.cat([all_gt_boxes, modified_boxes], dim=0)
                plotting_gts = torch.cat([plotting_gts,
                                          modified_boxes[0, 1:].reshape(1, -1)], dim=0)

            for class_num in range(num_classes):
                curr_class_scores = cls_probs[:, class_num]
                ####
                #curr_class_scores = torch.sigmoid(curr_class_scores)
                ###
                #score_threshold = 1 / cls_probs.shape[0]
                valid_score_idx = torch.where(curr_class_scores >= score_threshold)
                valid_scores = curr_class_scores[valid_score_idx]
                valid_proposals = proposals[valid_score_idx[0], :]
                #print(valid_proposals.shape, valid_scores.shape, proposals.shape)
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
                raw_image = tensor_to_PIL(bev[0].detach().cpu())

                for idx in range(plotting_proposals.shape[0]):
                    box_data = {"position": {
                        "minX": plotting_proposals[idx, 0].item() / dataset.req_img_size[0],
                        "minY": plotting_proposals[idx, 1].item() / dataset.req_img_size[0],
                        "maxX": plotting_proposals[idx, 2].item() / dataset.req_img_size[0],
                        "maxY": plotting_proposals[idx, 3].item() / dataset.req_img_size[0]},
                        "class_id": int(plotting_proposals[idx, 4].item()),
                        "box_caption": inv_class[int(plotting_proposals[idx][4])],
                        }
                    all_boxes.append(box_data)
                

                for idx in range(plotting_gts.shape[0]):
                    box_data_new = {"position": {
                        "minX": plotting_gts[idx, 1].item() / dataset.req_img_size[0],
                        "minY": plotting_gts[idx, 2].item() / dataset.req_img_size[0],
                        "maxX": plotting_gts[idx, 3].item() / dataset.req_img_size[0],
                        "maxY": plotting_gts[idx, 4].item() / dataset.req_img_size[0]},
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
                wandb.log({"Image proposals " + data['filename'][0]: box_image})
                # box_image = wandb.Image(raw_image, 
                #                         boxes= {"predictions":
                #                         {"box_data": all_gt_plotting_boxes,
                #                         "class_labels": inv_class}
                #                         })
                # wandb.log({"Image gt " + str(iter): box_image})
                
    for iou in iou_list:
        #print(all_gt_boxes.shape, all_gt_boxes.shape)
        AP = calculate_ap(all_pred_boxes, all_gt_boxes, iou, inv_class=inv_class, total_cls_num=num_classes)
        mAP = 0 if len(AP) == 0 else sum(AP) / len(AP)
        #return mAP.item(), AP
        wandb.log({"map@ " + str(iou): mAP})
        print("Iou ", iou, " mAP ", mAP)
    return mAP


if __name__ == '__main__':
    valid_data_list_filename = "./valid_full_list.txt"
    lidar_folder_name = "./data/"
    dataset = KITTICam(valid_data_list_filename=valid_data_list_filename, 
                            lidar_folder_name=lidar_folder_name)
    wandb.init("WSDNNResnet")
    epochs = 10
    model = WSDNN_Alexnet()

    train_dataset_length = int(0.80 * len(dataset))
    train_dataset, test_dataset = random_split(dataset, [train_dataset_length,
                                                        len(dataset) - train_dataset_length],
                                                        generator=torch.Generator().manual_seed(100))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(len(train_dataset), len(test_dataset))

    #scaler = torch.cuda.amp.GradScaler()
    loss_fn = nn.BCELoss(reduction='sum')
    #loss_fn = FocalLoss(alpha=0.25, gamma=2)
    #loss_fn = nn.MSELoss()
    model = model.cuda()
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.3)
    for i in range(epochs):
        # if i%1 == 0:
        #     model = model.eval()
        #     mAP = validate(test_loader, 
        #                   model, 
        #                   loss_fn, 
        #                   inv_class=dataset.inv_class, 
        #                   direct_class=dataset.class_to_int,
        #                   dataset=dataset)
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
                          direct_class=dataset.class_to_int,
                          dataset=dataset)

        scheduler.step()