from fileinput import filename
import numpy as np
import os
from tqdm import tqdm


def load_bbox(bboxfoldername,
              filename=None,
              pcl=False):
    if filename:
        bbox_filename = os.path.join(bboxfoldername, filename)
    else:
        bbox_filename = bboxfoldername

    boxes = np.empty((0, 6))
    with open(bbox_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if pcl:
                # a "\n" is present
                A = line[:-2].split(',')
            else:
                A = line.split(',')
            B = np.array([float(a) for a in A])
            boxes = np.vstack([boxes, B.reshape(1, -1)])
    return boxes

def filter_boxes(boxes,
                 box_no_req,
                 area_min=0.10,
                 bev_xmax=70,
                 bev_ymin=-40,
                 bev_ymax=40):

    valid_box_ids = np.where(boxes[:, 3] <= bev_xmax)
    valid_boxes = boxes[valid_box_ids]
    valid_box_ids = np.where(boxes[:, 1] >= bev_ymin)
    valid_boxes = valid_boxes[valid_box_ids]
    valid_box_ids = np.where(boxes[:, 4] <= bev_ymax)
    valid_boxes = valid_boxes[valid_box_ids]

    if box_no_req > valid_boxes.shape[0]:
        return np.empty((0))

    volume = (valid_boxes[:, 5] - valid_boxes[:, 2]) * \
             (valid_boxes[:, 4] - valid_boxes[:, 1]) * \
             (valid_boxes[:, 3] - valid_boxes[:, 0])

    volume_valid_ids = np.where(volume >= area_min)
    volume_valid = volume[volume_valid_ids]
    valid_volume_boxes = valid_boxes[volume_valid_ids]
    
    # if vol filter gives less number of boxes, give the last box_no_req
    # sorted by volume
    if valid_volume_boxes.shape[0] < box_no_req:
        #print("not enough area filtered boxes")
        volume_ascending_ids = np.argsort(-volume)
        volume_ascending = volume[volume_ascending_ids]
        valid_volume_boxes = valid_boxes[volume_ascending_ids]
        return valid_volume_boxes[:box_no_req]
    
    # else return the first box_no_req boxes, we do not need high volume boxes
    # might be walls etc
    else:
        #print("enough area filetered boxes")
        volume_sort_args = np.argsort(volume_valid)
        volume_valid = volume_valid[volume_sort_args]
        valid_volume_boxes = valid_volume_boxes[volume_sort_args]
        return valid_volume_boxes[:box_no_req]  

def data_prepare(basefilename="/media/akshay/Data/KITTI/",
                 bbox_filename="bbox",
                 bbox_pcl_filename="bbox_pcl",
                 training=True,
                 no_of_boxes=100):
    if training:
        data_path = os.path.join(basefilename, "training")
    else:
        data_path = os.path.join(basefilename, "testing")
    
    data_bbox_path = os.path.join(data_path, bbox_filename)
    data_bbox_pcl_path = os.path.join(data_path, bbox_pcl_filename)

    filename_list = [file_name for file_name in os.listdir(data_bbox_path)]
    bad_proposals = 0

    with open("valid_filenames_test.txt", "w") as f:
        for file_name in tqdm(os.listdir(data_bbox_path)):
            file_idx = file_name.split(".")[0]
            pcl_bbox = load_bbox(data_bbox_pcl_path, file_name, True)
            bbox = load_bbox(data_bbox_path, file_name, False)
            valid_box_req = filter_boxes(bbox, no_of_boxes - pcl_bbox.shape[0])
            if valid_box_req.shape[0] == 0:
                print("not enough")
                bad_proposals += 1
            else:
                print(pcl_bbox.shape, valid_box_req.shape)
                f.write(file_idx + "\n")
        print(bad_proposals, " failed")


if __name__ == "__main__":
    data_prepare(training=False)