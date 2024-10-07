import aprilgrid
import cv2
import numpy as np
import argparse
import os
import yaml
from typing import List
from configparase import CameraCV, Camera
from utils.constant import CALIB_POS_TO_CAM_MAP, CALIB_BOARD_PARAMS


def solve_pnp(obj_points, img_points, camera:Camera):
    ret, rvec, tvec = cv2.solvePnP(
        np.array(obj_points), np.array(img_points), camera.cameraMatrix, camera.distCoeffs)

    return rvec, tvec

class AprilgridDetector:
    def __init__(self, families: List[str]):
        self.detectors = {
            family: aprilgrid.Detector(tag_family_name=family) for family in families
        }

    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = {}
        for family, detector in self.detectors.items():
            res = detector.detect(gray)
            if len(res) != 0:
                result[family] = res
        return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cams", required=True, nargs='+', help="selected cams")
    parser.add_argument("--calib_dirs", required=True, help="path to the intrinsic of cameras")
    parser.add_argument("--record", required=True, help="path to record")
    return parser.parse_args()


def match_and_compute(image_infos_1, image_infos_2):
    """
    """

    for (stamp_1, rvec_1, tvec_1) in image_infos_1:
        for (stamp_2, rvec_2, tvec_2) in image_infos_2:
            T_cam2_2_cam1 = np.eye(4, dtype=np.float32)
            T_cam2_2_cam1[:3, :3] = cv2.Rodrigues(rvec_1)[0] @ cv2.Rodrigues(rvec_2)[0].T
            T_cam2_2_cam1[:3, 3:] = tvec_1 - T_cam2_2_cam1[:3,:3] @ tvec_2
            return T_cam2_2_cam1



def main(args):
    """
    """
    if len(args.cams) <= 1:
        print(f"cams required at least two.")
        return
    if not os.path.exists(args.record):
        print(f"record doesn't exist.")
        return
    tag_detector = AprilgridDetector(['t16h5'])

    cams_images = {}
    for cam in args.cams:
        cam_dir = os.path.join(args.record, cam)
        if not os.path.exists(cam_dir):
            continue
        info = []
        cam_calib_file = os.path.join(args.calib_dirs, f"{cam}.yaml")
        print(f"cam_calib_files... {cam_calib_file}, {os.path.exists(cam_calib_file)}")
        cam_intrinsic = Camera(os.path.abspath(cam_calib_file))
        for file in os.listdir(cam_dir):
            if file.endswith(".jpg"):
                image_path = os.path.join(cam_dir, file)
                image = cv2.imread(image_path)
                tag_results = tag_detector.detect(image)
                if len(tag_results) != 0:
                    print(f"find tags in {image_path}")
                    img_points = []
                    obj_points = []
                    for tag_family, tags in tag_results.items():
                        for tag in tags:
                            img_points.extend(tag.corners[:, 0,:].tolist())
                            obj_points.extend(CALIB_BOARD_PARAMS[tag_family]['corners'].reshape(-1, 4, 3)[tag.tag_id].tolist())
                    rvec1, tvec1 = solve_pnp(obj_points, img_points, cam_intrinsic)
                    tr_calib_board_to_cam = np.eye(4, dtype=np.float32)
                    tr_calib_board_to_cam[:3, :3] = cv2.Rodrigues(rvec1)[0]
                    tr_calib_board_to_cam[:3, 3:] = tvec1
                    print(f"tr_calib_board_to_cam {cam}, {tr_calib_board_to_cam}")
                    word = file.split(".")[:-1]
                    if len(word) == 1:
                        sec = int(word[0])
                        nano_sec = int(word[0]) % 1e9
                    elif len(word) > 1:
                        sec = int(word[0])
                        nano_sec = int(word[1])
                    stamp = sec + nano_sec
                    info.append((stamp, rvec1, tvec1))
        cams_images[cam] = info
    # sort by stamps
    for cam_1 in args.cams:
        image_infos_1 = cams_images[cam_1]
        if len(image_infos_1) == 0:
            continue
        for cam_2 in args.cams:
            image_infos_2 = cams_images[cam_2]
            if cam_1 == cam_2:
                continue
            if len(image_infos_2) == 0:
                continue

            Tr_cam2_2_cam1 = match_and_compute(image_infos_1, image_infos_2)
            print(f"cam {cam_2} to {cam_1} is {Tr_cam2_2_cam1}")


if __name__ == '__main__':
    # get image path from command line
    main(parse_args())
