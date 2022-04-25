import numpy as np
import os  

def scale_bev(bev, map_height=800):
    bev_new = bev / 0.1
    bev_new[:, 0] += int(map_height // 2)
    bev_new[:, 2] += int(map_height // 2)
    # bev_new[:, 0] = map_height - bev_new[:, 0]
    # bev_new[:, 2] = map_height - bev_new[:, 2]
    return bev_new

def get_gt_bbox(bbox):
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
    bev = scale_bev(bev)
    return bev

if __name__ == '__main__':
    folder_name = "/media/akshay/Data/KITTI/training/label_2/"
    output_file_list = "valid_data_list_after_threshold.txt"

    valid_data_list_filename = "valid_filenames.txt"
    filenames_list = []
    with open(valid_data_list_filename, "r") as f: 
            for line in f.readlines():
                line = line.split("\n")[0]
                filenames_list.append(line)

    print(len(filenames_list))
    count = 0
    for iter, filename in enumerate(filenames_list):
        print(filename)
        filename_full = os.path.join(folder_name, filename + ".txt")

        min_now = 10000
        with open(filename_full, "r") as f:
            for line in f.readlines():
                words = line.split(" ")
                if words[0] == 'DontCare':
                    continue
                else:
                    curr_box_labels = [float(words[i]) for i in range(8, 15)]
                    corners = get_gt_bbox(curr_box_labels)
                    min_now = min(min_now, corners[0, 1])
                    #print(corners[0, 1], min_now)

            if min_now == 10000:
                continue
            elif min_now >= 300:
                continue
            else:
                count += 1
                with open(output_file_list, "a") as out_file:
                    out_file.write(filename + "\n")
        
    print(count)