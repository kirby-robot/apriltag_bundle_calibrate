import os.path

import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml
from calib import Camera
from pypcd4 import PointCloud


def load_yaml(fname, encoding='utf-8'):
    with open(fname, "r", encoding=encoding) as f:
        content = yaml.load(f, yaml.SafeLoader)
        return content


def save_yaml(data, fname, encoding='utf-8'):
    with open(fname, "w", encoding=encoding) as f:
        yaml.dump(data, f)


def load_lidar(root, lidar_dir, tr):
    pcd_all = [PointCloud.from_path(root + "/" + lidar_dir + "/" + file).numpy()
               for file in os.listdir(root + "/" + lidar_dir)[:30]]
    pcd_tr = []
    for pcd in pcd_all:
        xyz = (tr[:3, :3] @ pcd[:, :3].T + tr[:3, 3:]).T
        d = np.concatenate([xyz, pcd[:, 3:]], axis=1)
        pcd_tr.append(d)
    pcd_all = np.vstack(pcd_tr)

    return pcd_all


LIDAR_TOPICS = ['/lidar_192_168_1_154', '/lidar_192_168_1_166']


def load_scene_lidar(root):
    # print(f"root... {root}")
    decompress_dirs = [f for f in os.listdir(root) if f.find("_dump") != -1]
    folder = decompress_dirs[0]
    # print(f"folder ... {folder}")
    pcd_1 = load_lidar(os.path.join(root, folder), LIDAR_TOPICS[0], tr=np.eye(4))
    pcd_2 = load_lidar(os.path.join(root, folder), LIDAR_TOPICS[1], tr=np.linalg.inv(Tr_lidar_154_to_166))

    return np.vstack([pcd_1, pcd_2])


def find_decompress_dir(root):
    decompress_dir = [f for f in os.listdir(root) if f.find("_dump") != -1]
    return os.path.join(root, decompress_dir[0])


data = load_yaml("lidar_calib.yaml")

print(f"data... {data.keys()}")

root = "/home/anchen/solutions/data"

Tr_lidar_154_to_166 = np.array([[0.99868202, 0.04797583, 0.01823565, -0.95085806],
                                [0.04873902, -0.99784159, -0.04400752, 9.37028002],
                                [0.01608499, 0.04483831, -0.99886476, 7.28877078],
                                [0., 0., 0., 1.]])

CAMERAS = ["image_info_1", "image_info_2", "image_info_3", "image_info_4", "image_info_5", "image_info_6",
           "image_info_7"]

camera_infos = []
calib_root = os.path.join(root, "calib_files")
cam_dicts = {}
for cam in CAMERAS:
    cam_calib_file = os.path.join(calib_root, f"{cam}.yaml")
    camera = Camera.load(cam_calib_file)
    camera_infos.append({
        "name": cam,
        "topic": f"/{cam}",
        "D": camera.D,
        "M": camera.M,
        "P": camera.P,
        "width": camera.width,
        "height": camera.height
    })
    cam_dicts[cam] = camera

sensor_calibrations = {}

for scene in data.keys():
    path_to_board = os.path.join(root, scene, "board_to_cam.yaml")
    print(f"scene... {scene}")
    if os.path.exists(path_to_board):
        board_to_cams = load_yaml(path_to_board)
        lidar_to_board = data[scene]
        lidar_to_board = np.array(lidar_to_board).reshape(4, 4)
        scene_pointcloud = load_scene_lidar(os.path.join(root, scene))
        for cam in board_to_cams.keys():
            board_to_cam = np.array(board_to_cams[cam]).reshape(4, 4)
            lidar_to_cam = board_to_cam @ lidar_to_board
            # print(f"lidar 154 to cam {cam} : {lidar_to_cam}")
            if cam not in sensor_calibrations:

                cam_dir = os.path.join(find_decompress_dir(os.path.join(root, scene)), cam)
                image_files = os.listdir(cam_dir)
                print(f"image_files[0]... {image_files[0]}")
                image = cv2.imread(os.path.join(cam_dir, image_files[0]))
                assert isinstance(cam_dicts[cam], Camera)
                points = scene_pointcloud[:, :3]
                mask = scene_pointcloud[:, 3] > 140
                if cam in ['image_info_2']:
                    lidar_to_cam2 = np.array([[-0.04698819, -0.97387313, 0.22217844, 3.62442765],
                                             [0.78428455, -0.17371672, -0.5955839, -0.99823368],
                                             [0.61861925, 0.14626574, 0.77195625, -1.6969182],
                                             [0., 0., 0., 1.]])
                    print(f"check image_info_2.... {lidar_to_cam @ np.linalg.inv(lidar_to_cam2)}")
                    lidar_to_cam = lidar_to_cam2
                # if cam in ['image_info_2', 'image_info_3', 'image_info_4']:
                # tr_reverse = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
                # lidar_to_cam = tr_reverse @ lidar_to_cam
                points_in_cam = (lidar_to_cam[:3, :3] @ points.T + lidar_to_cam[:3, 3:]).T
                mask = (points_in_cam[:, 2] > 0) & (scene_pointcloud[:, 3] > 0)
                points_in_cam = points_in_cam[mask]
                depth = points_in_cam[:, 2]
                uvs = cam_dicts[cam].projectPoints(points_in_cam)
                x = []
                y = []
                for pt in uvs:
                    x_2d, y_2d = pt[0], pt[1]
                    if 0 <= x_2d < image.shape[1] and 0 <= y_2d < image.shape[0]:
                        x.append(x_2d)
                        y.append(y_2d)
                x = np.array(x)
                y = np.array(y)
                plt.scatter(x, y, s=1)
                plt.imshow(image)
                plt.show()
                print(f"save... {cam}")
                sensor_calibrations[cam] = lidar_to_cam

calibration_topos = [{"source_frame": "lidar_154",
                      "target_frame": "lidar_166",
                      "extrinsic": Tr_lidar_154_to_166},
                     {"source_frame": "base_frame",
                      "target_frame": "lidar_154",
                      "extrinsic": np.eye(4, dtype=np.float32)}]

for cam in sensor_calibrations:
    calibration_topos.append({"source_frame": "lidar_154",
                              "target_frame": cam,
                              "extrinsic": sensor_calibrations[cam]})

room_calib = {
    "sensors": {
        "cameras": camera_infos,
        "lidars": [
            {"name": "lidar_154",
             "topic": "/livox/192.168.1.154"},
            {"name": "lidar_166",
             "topic": "/livox/192.168.1.166"}
        ]
    },
    "calibrations": calibration_topos
}

save_yaml(room_calib, f"{root}/room_calibration.yaml")
