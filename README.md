# Weakly Supervised Lidar Object Detection
## This Repo contains our custom model that performs weakly supervised object detection using Proposals generataed by density based clustering
Our code uses KITTI 3D object detection dataset. Our proposals are availiable in data folder
To run the file change lidar_folder_name in main_resnet.py to your preferred location. Default data folder
To train and validate our Weakly Supervised Lidar Object detection Model Run  
``` python3 main_resnet.py ```  
Pretrained weights for resnet152 based encoder is availiable in model_res_focal_loss_plt.pth
