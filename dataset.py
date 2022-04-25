from matplotlib.lines import Line2D
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os  
from dataprocess import load_bbox, filter_boxes
import ctypes
from tqdm import tqdm

class KITTIBEV(Dataset):
    def __init__(self, 
                is_train=True,
                lidar_folder_name=None,
                label_folder_name=None,
                valid_data_list_filename=None):

        self.class_names = ['Car', 
                            'Van', 
                            'Truck',
                            'Pedestrian', 
                            'Person_sitting', 
                            'Cyclist', 
                            'Tram',
                            'Misc']

        self.geometry = {'L1': -40.0,
                        'L2': 40.0,
                        'W1': 0.0,
                        'W2': 70.0,
                        'H1': -2.5,
                        'H2': 1.0,
                        'input_shape': (800, 700, 36),
                        'label_shape': (200, 175, 7)}

        self.use_npy = False
        self.num_classes = len(self.class_names)
        self.lidar_folder_name = lidar_folder_name
        self.label_folder_name = label_folder_name
        self.LidarLib = ctypes.cdll.LoadLibrary('./preprocess/LidarPreprocess.so')
        self.is_train = is_train
        if self.is_train:
            self.sub_folder = 'training'
        else:
            self.sub_folder = 'testing'
        self.filenames_list = []

        with open(valid_data_list_filename, "r") as f: 
            for line in f.readlines():
                line = line.split("\n")[0]
                self.filenames_list.append(line)
        
        self.inv_class = {i: class_name for i, class_name in enumerate(self.class_names)}
        self.class_to_int = {class_name: i for i, class_name in enumerate(self.class_names)}

        ####
        self.preload_proposals = []
        self.preload_labels = []
        self.preload_gt_boxes = []
        self.preload_gt_class_list = []

        print("Preloading Data")
        for filename in tqdm(self.filenames_list,
                            total=len(self.filenames_list),
                            leave=False):
            proposals = self.get_proposals(filename)
            self.preload_proposals.append(proposals)
            labels, gt_boxes, gt_class_list = self.get_labels(filename)
            self.preload_labels.append(labels)
            self.preload_gt_boxes.append(gt_boxes)
            self.preload_gt_class_list.append(gt_class_list)
        ####

    def __len__(self) -> int:
        return len(self.filenames_list)

    def __getitem__(self, index: int):
        filename = self.filenames_list[index]
        bev = self.load_velo_scan(index)
        #bev = self.lidar_preprocess(bev)
        bev = bev.transpose(2, 0, 1)
        proposals = self.preload_proposals[index]
        labels = self.preload_labels[index] 
        gt_boxes = self.preload_gt_boxes[index]
        gt_class_list = self.preload_gt_class_list[index] # .get_labels(self.filenames_list[index])
        # print(labels.shape, proposals.shape, gt_boxes.shape, gt_class_list.shape)
        # print(len(self.preload_proposals),
        #       len(self.preload_labels),
        #       len(self.preload_gt_boxes),
        #       len(self.preload_gt_class_list))
        return {'bev': torch.from_numpy(bev),
                'labels': torch.from_numpy(labels),
                'gt_boxes': torch.from_numpy(gt_boxes),
                'proposals': torch.from_numpy(proposals),
                'gt_class_list': torch.from_numpy(gt_class_list)}

    def get_proposals(self, filename):
        pcl_proposal_filename = os.path.join(self.lidar_folder_name,
                                             self.sub_folder,
                                             "bbox_pcl",
                                             filename + ".txt")
        dbscan_proposal_filename = os.path.join(self.lidar_folder_name,
                                                self.sub_folder,
                                                "bbox",
                                                filename + ".txt")
        pcl_boxes = load_bbox(pcl_proposal_filename, pcl=True)
        pcl_boxes[:, 1] = np.where(pcl_boxes[:, 1] < -40, -40, pcl_boxes[:, 1]) #np.max(pcl_boxes[:, 2], -40)
        pcl_boxes[:, 3] = np.where(pcl_boxes[:, 3] > 70, 70, pcl_boxes[:, 3]) #np.min(pcl_boxes[:, 3], 70)
        pcl_boxes[:, 4] = np.where(pcl_boxes[:, 4] > 40, 40, pcl_boxes[:, 4]) #np.min(pcl_boxes[:, 4], 40)

        dbscan_boxes = load_bbox(dbscan_proposal_filename, pcl=False) 
        dbscan_filtered_boxes = filter_boxes(dbscan_boxes,
                                            100 - pcl_boxes.shape[0])
        combined_boxes = np.concatenate([pcl_boxes, dbscan_filtered_boxes], axis=0)

        # check row order
        combined_boxes_2d = np.concatenate([combined_boxes[:, 1].reshape(-1, 1),
                                            combined_boxes[:, 0].reshape(-1, 1), 
                                            combined_boxes[:, 4].reshape(-1, 1), 
                                            combined_boxes[:, 3].reshape(-1, 1)], axis=1)

        ##### box augmentation
        augment_distance = (combined_boxes[:,3] - combined_boxes[:, 0]).reshape(-1, 1) * 2 # N*1
        new_y_max = combined_boxes[:, 0].reshape(-1, 1) + augment_distance
        new_y_max = np.where(new_y_max[:, 0] >= 70, 69, new_y_max[:, 0])
        #####
        combined_augmented_boxes = np.concatenate([combined_boxes[:, 1].reshape(-1, 1),
                                                   combined_boxes[:, 0].reshape(-1, 1), 
                                                   combined_boxes[:, 4].reshape(-1, 1),
                                                   new_y_max.reshape(-1, 1)], axis=1)
        combined_augmented_boxes_2d = np.concatenate([combined_boxes_2d,
                                                     combined_augmented_boxes], axis=0)
        combined_augmented_boxes_2d = self.scale_bev(combined_augmented_boxes_2d)
        #####
        return combined_augmented_boxes_2d
    
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
    
    def point_in_roi(self, point):
        if (point[0] - self.geometry['W1']) < 0.01 or (self.geometry['W2'] - point[0]) < 0.01:
            return False
        if (point[1] - self.geometry['L1']) < 0.01 or (self.geometry['L2'] - point[1]) < 0.01:
            return False
        if (point[2] - self.geometry['H1']) < 0.01 or (self.geometry['H2'] - point[2]) < 0.01:
            return False
        return True

    def get_labels(self, filename):
        label = np.zeros((self.num_classes,))
        gt_boxes = np.zeros((0, 4))
        gt_class_list = []
        label_filename = os.path.join(self.lidar_folder_name,
                                      self.sub_folder,
                                      "label_2", 
                                      filename + ".txt")
        with open(label_filename, "r") as f:
            for line in f.readlines():
                x = line.split(" ")
                if x[0] == 'DontCare':
                    continue
                label[self.class_to_int[x[0]]] = 1
                gt_class_list.append(self.class_to_int[x[0]])
                curr_box_labels = [float(x[i]) for i in range(8, 15)]
                gt_box_curr = self.get_gt_bbox(curr_box_labels)
                if gt_box_curr[0, 1] >= 300:
                    continue
                gt_boxes = np.concatenate([gt_boxes, gt_box_curr], axis=0)
        return label, gt_boxes, np.asarray(gt_class_list).reshape(-1)

    def scale_bev(self, bev, map_height=800):
        bev_new = bev / 0.1
        bev_new[:, 0] += int(map_height // 2)
        bev_new[:, 2] += int(map_height // 2)
        # bev_new[:, 0] = map_height - bev_new[:, 0]
        # bev_new[:, 2] = map_height - bev_new[:, 2]
        return bev_new

    def get_gt_bbox(self, bbox):
        w, h, l, y, z, x, yaw = bbox
        y = -y
        yaw = -(yaw + np.pi / 2)
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        # rear left
        bev_corners[0, 0] = x - l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[0, 1] = y - l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        # rear right
        bev_corners[1, 0] = x - l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[1, 1] = y - l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front right
        bev_corners[2, 0] = x + l/2 * np.cos(yaw) + w/2 * np.sin(yaw)
        bev_corners[2, 1] = y + l/2 * np.sin(yaw) - w/2 * np.cos(yaw)

        # front left
        bev_corners[3, 0] = x + l/2 * np.cos(yaw) - w/2 * np.sin(yaw)
        bev_corners[3, 1] = y + l/2 * np.sin(yaw) + w/2 * np.cos(yaw)

        max_x = np.max(bev_corners[:, 0])
        min_x = np.min(bev_corners[:, 0])
        max_y = np.max(bev_corners[:, 1])
        min_y = np.min(bev_corners[:, 1])

        bev = np.array([min_y, min_x, max_y, max_x]).reshape(-1, 4)
        bev = self.scale_bev(bev)
        return bev

    def load_velo_scan(self, index):
        """Helper method to parse velodyne binary files into a list of scans."""
        filename = os.path.join(self.lidar_folder_name,
                                self.sub_folder,
                                "velodyne", 
                                self.filenames_list[index] + ".bin")

        if self.use_npy:
            scan = np.load(filename[:-4]+'.npy')
        else:
            c_name = bytes(filename, 'utf-8')
            scan = np.zeros(self.geometry['input_shape'], dtype=np.float32)
            c_data = ctypes.c_void_p(scan.ctypes.data)
            self.LidarLib.createTopViewMaps(c_data, c_name)
            #scan = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
            
        return scan

if __name__ == "__main__":
    valid_data_list_filename = "valid_filenames.txt"
    lidar_folder_name = "/media/akshay/Data/KITTI/"
    dataset = KITTIBEV(valid_data_list_filename=valid_data_list_filename, 
                       lidar_folder_name=lidar_folder_name)
    dataset[10]