import os.path

import numpy as np
import cv2
import yaml
from calib import Camera


def load_yaml(fname, encoding='utf-8'):
    with open(fname, "r", encoding=encoding) as f:
        content = yaml.load(f, yaml.SafeLoader)
        return content


def save_yaml(data, fname, encoding='utf-8'):
    with open(fname, "w", encoding=encoding) as f:
        yaml.dump(data, f)


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

sensor_calibrations = {}

for scene in data.keys():
    path_to_board = os.path.join(root, scene, "board_to_cam.yaml")
    if os.path.exists(path_to_board):
        board_to_cams = load_yaml(path_to_board)
        lidar_to_board = data[scene]
        lidar_to_board = np.array(lidar_to_board).reshape(4, 4)
        for cam in board_to_cams.keys():
            board_to_cam = np.array(board_to_cams[cam]).reshape(4, 4)
            lidar_to_cam = board_to_cam @ lidar_to_board
            print(f"lidar 154 to cam {cam} : {lidar_to_cam}")
            if cam not in sensor_calibrations:
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


def load_lidars(root, lidars):
    """

    """

