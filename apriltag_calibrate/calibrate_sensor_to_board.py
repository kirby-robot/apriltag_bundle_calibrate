import aprilgrid
import cv2

import numpy as np

import argparse

import os
from utils.constant import CALIB_BOARD_PARAMS, CALIB_POS_TO_CAM_MAP
from configparase import Camera


class AprilgridDetector():
    def __init__(self, families):
        self._detectors = {
            family: aprilgrid.Detector(tag_family_name=family) for family in families
        }

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = {}
        for family, detector in self._detectors.items():
            res = detector.detect(gray)
            if len(res) != 0:
                result[family] = res
        return result


def calibrate_camera(camera_image_dir, camera_calib_path, detector):
    camera = Camera(camera_calib_path)
    img_points = []
    obj_points = []
    detected_families = set()
    for file in os.listdir(camera_image_dir):
        if not file.endswith(".jpg"):
            continue
        image = cv2.imread(os.path.join(camera_image_dir, file))
        tag_results = detector.detect(image)
        if len(tag_results) < 1:
            continue
        for tag_family, tags in tag_results.items():
            for tag in tags:
                if tag.tag_id < len(CALIB_BOARD_PARAMS[tag_family]['corners'].reshape(-1, 4, 3)):
                    obj_points.extend(CALIB_BOARD_PARAMS[tag_family]['corners'].reshape(-1, 4, 3)[tag.tag_id].tolist())
                    img_points.extend(tag.corners[:, 0, :].tolist())
                    detected_families.add(tag_family)

    if len(detected_families) < 1:
        return False, None

    ret, rvec, tvec = cv2.solvePnP(
        np.array(obj_points), np.array(img_points), camera.cameraMatrix, camera.distCoeffs)
    if ret is False:
        return False, None

    tr_calib_board_to_cam = np.eye(4, dtype=np.float32)
    tr_calib_board_to_cam[:3, :3] = cv2.Rodrigues(rvec)[0]
    tr_calib_board_to_cam[:3, 3:] = tvec

    print(f"tr_calib_board_to_cam .... {tr_calib_board_to_cam}")

    return True, tr_calib_board_to_cam


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="path to dump directory")
    parser.add_argument("--calib_dir", required=True, help="path to calibration directory")
    parser.add_argument("--cams", default=[], nargs='+', help="selected cameras")
    parser.add_argument("--lidars", default=[], nargs='+', help="selected lidars.")
    return parser.parse_args()


def main(args):
    tag_detector = AprilgridDetector(['t16h5', 't25h7', 't36h11'])
    sensor_topos = {}

    for cam in args.cams:
        cam_dir = os.path.join(args.root, cam)
        if not os.path.exists(cam_dir):
            continue
        cam_calib_file = os.path.join(args.calib_dir, f"{cam}.yaml")

        ret, tr = calibrate_camera(cam_dir, cam_calib_file, tag_detector)
        if not ret:
            print(f"failed to calibrate {cam}")
            continue
        print(f"success calibrate {cam} tr_board to_cam {tr}")
        sensor_topos[cam] = tr

    for lidar in args.lidars:
        pass


if __name__ == '__main__':
    main(parse_args())
