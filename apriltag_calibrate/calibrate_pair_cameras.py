import aprilgrid
import cv2
import numpy as np
import argparse
import os
import yaml
from typing import List
from configparase import CameraCV
from utils.constant import CALIB_POS_TO_CAM_MAP, CALIB_BOARD_PARAMS


def solve_pnp(obj_points, img_points, camera: CameraCV):
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
        result = {
            family: detector.detect(gray) for family, detector in self.detectors.items()
        }
        return result

if __name__ == '__main__':
    # get image path from command line
    ap = argparse.ArgumentParser()
    ap.add_argument("--img1", required=True, help="path to the image1")
    ap.add_argument("--intrinsic1", required=True, help="path to the intrinsic of cam1")
    ap.add_argument("--img2", required=True, help="path to the image2")
    ap.add_argument("--intrinsic2", required=True, help="path to the intrinsic of cam2")
    ap.add_argument("--output", required=True, help="path to save the result")
    args = ap.parse_args()

    tag_detector = AprilgridDetector(['t36h11'])    # 't16h5', 't25h7',

    print(f'Start process {args.img1}')
    img_points = []
    obj_points = []
    image1 =cv2.imread(args.img1)
    detect_results1 = tag_detector.detect(image1)
    for tag_family, tags in detect_results1.items():
        print(f'detect {len(tags)} {tag_family} tags in img1...')
        for tag in tags:
            img_points.extend(tag.corners[:, 0, :].tolist())
            obj_points.extend(
                CALIB_BOARD_PARAMS[tag_family]['corners'].reshape(-1, 4, 3)[tag.tag_id].tolist()
            )
    print(f'add {len(img_points)} image points, {len(obj_points)} objec points')
    rvec1, tvec1 = solve_pnp(obj_points, img_points, CameraCV(args.intrinsic1))

    print(f'Start process {args.img2}...')
    img_points = []
    obj_points = []
    image2 =cv2.imread(args.img2)
    detect_results2 = tag_detector.detect(image2)
    for tag_family, tags in detect_results2.items():
        print(f'detect {len(tags)} {tag_family} tags in img2...')
        for tag in tags:
            img_points.extend(tag.corners[:, 0, :].tolist())
            obj_points.extend(
                CALIB_BOARD_PARAMS[tag_family]['corners'].reshape(-1, 4, 3)[tag.tag_id].tolist()
            )
    print(f'add {len(img_points)} image points, {len(obj_points)} objec points')
    rvec2, tvec2 = solve_pnp(obj_points, img_points, CameraCV(args.intrinsic2))

    T_cam2_2_cam1 = np.eye(4, dtype=np.float)
    T_cam2_2_cam1[:3,:3] = cv2.Rodrigues(rvec1)[0] @ cv2.Rodrigues(rvec2)[0].T
    T_cam2_2_cam1[:3, 3:] = tvec1 - T_cam2_2_cam1[:3,:3] @ tvec2

    print(f'rvec1: {rvec1}, tvec1: {tvec1}, rvec2: {rvec2}, tvec2: {tvec2}')
    print(f'T_cam2_2_cam1: \n{T_cam2_2_cam1}')
