import aprilgrid
import cv2
import numpy as np
import argparse
import os
import yaml
from typing import List
from configparase import Camera
from concurrent.futures import ProcessPoolExecutor
from bundle_calibrate import (
    WarmupPoseGraph,
    BundleCalibratePoseGraph,
    BundleImageLoader,
    solve_pnp,
)
from functools import partial
from tqdm import tqdm
from utils.constant import CALIB_POS_TO_CAM_MAP, CALIB_BOARD_PARAMS

class MultiCamImageLoader:
    '''
    Params:
        path: input images path
            folder structure:
                path
                ├── calib_pos0
                |   ├── cam0_image_000.jpg
                |   ├── cam1_image_000.jpg
                |   ├── ...
                |   ├── cam6_image_000.jpg
                ├── calib_pos1
                |   ├── cam0_image_000.jpg
                |   ├── cam1_image_000.jpg
                |   ├── ...
                |   ├── cam6_image_000.jpg
                ├── ...
                ├── calib_pos3
                |   ├── cam0_image_000.jpg
                |   ├── cam1_image_000.jpg
                |   ├── ...
                |   ├── cam6_image_000.jpg
        use_mp: whether to use multi processing
    '''
    def __init__(self, path: str, use_mp=False) -> None:
        self.path = path
        self.use_mp = use_mp
        self.images = dict()

    def load_img(self, folder_path, file_path):
        if not file_path.endswith(".jpg"):
            return False, "", None
        image_path = os.path.join(folder_path, file_path)
        image = cv2.imread(image_path)
        return image is not None, file_path.split("/")[-1], image

    def load_bundle(self, folder, files):
        bundle = []
        print("reading images...")
        with ProcessPoolExecutor() as executor:
            if self.use_mp:
                images = list(
                    tqdm(executor.map(partial(self.load_img, folder), files), total=len(files)))
            else:
                images = [self.load_img(folder, file) for file in files]

            for img in images:
                if img[0]:
                    bundle.append((img[1], img[2]))
        return bundle

    def load(self):
        for folder, _, files in os.walk(self.path):
            bundle = self.load_bundle(folder, files)
            self.images.setdefault(folder.split('/')[-1], bundle)

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
    ap.add_argument("--input", required=True, help="path to the images")
    ap.add_argument("--output", required=True, help="path to save the result")
    ap.add_argument("--intrinsic", required=True, help="path to camera calib files")
    ap.add_argument("--cameras", default="cam0,cam1", help="camera sensor names")
    args = ap.parse_args()

    cam_keys = args.cameras.split(',')
    cameras = {cam: Camera(os.path.join(args.intrinsic, cam + '.yaml')) for cam in cam_keys}

    pose_graph = WarmupPoseGraph()
    tag_detector = AprilgridDetector(['t16h5', 't25h7', 't36h11'])
    image_load = MultiCamImageLoader(args.input, cam_keys, use_mp=True)
    image_load.load()
    for calib_pos, images in image_load.images.items():
        bundle_key = pose_graph.add_bundle()
        print(f'start process {calib_pos}...')
        for file, image in tqdm(images):
            cam_name = file.split('_')[0]
            if cam_name not in CALIB_POS_TO_CAM_MAP[calib_pos]:
                # skip specify cameras for every calib_pos
                continue
            img_points = []
            obj_points = []
            detect_results = tag_detector.detect(image)
            for tag_family, tags in detect_results.items():
                for tag in tags:
                    img_points.extend(tag.corners[:, 0, :].tolist())
                    obj_points.extend(
                        CALIB_BOARD_PARAMS[tag_family]['corners'].reshape(-1, 4, 2)[tag.tag_id].tolist()
                    )
            rvec, tvec = solve_pnp(obj_points, img_points, cameras[cam_name])
            camera_key = pose_graph.add_camera()
            # Not complete
            pose_graph.add_tag()
