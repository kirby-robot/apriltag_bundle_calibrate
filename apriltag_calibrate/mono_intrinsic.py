import cv2
import numpy as np
import argparse
import yaml
import os
import apriltag
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from apriltag_calibrate.utils.constant import CALIB_BOARD_PARAMS
import aprilgrid


# get image path from command line
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-r", "--root", required=True,
                    help="folder path to calibration data folder.")
    ap.add_argument("--camera", required=True, help="camera to calibrate.")
    # ap.add_argument("-t", "--tag", required=True, help="tag config file")
    return ap.parse_args()


def load_all_images(folder_path):
    # read all the image to a list
    image_files = os.listdir(folder_path)
    for file in image_files:
        if not file.endswith(".jpg") and not file.endswith(".png"):
            image_files.remove(file)

    def load_img(file):
        image_path = os.path.join(folder_path, file)
        image = cv2.imread(image_path)
        if image is not None:
            return image

    print(f"reading images... {len(image_files)}")
    images = [os.path.join(folder_path, fname) for fname in image_files]
    return images


class AprilgridDetector:
    def __init__(self, families):
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


def build_detector():
    tag_families = ['t25h7', 't36h11', 't16h5']
    detector = AprilgridDetector(tag_families)
    return detector


def main(args):
    print("reading apriltag config...")
    detector = build_detector()
    vis = False
    print("process img...")

    img_points = []
    obj_points = []
    init_intrinsic = np.array([[1316.43, 0, 1521.59],
                               [0, 1307.11, 1020.1643],
                               [0, 0, 1]])
    init_distortion = np.array([-0.02889376, -0.04168648, 0.00028455, -0.00336097, 0.01589657])
    img_size = (3072, 2048)
    for folder in ['p1', 'p2', 'p3', 'p4', 'p5', 'p6']:
        path_to_folder = os.path.join(args.root, folder)
        for sub_dir in os.listdir(os.path.join(path_to_folder)):
            if sub_dir.find("dump") != -1:
                path_to_folder = os.path.join(path_to_folder, sub_dir)
        print(f"path_to_folder ... {path_to_folder}")
        images = load_all_images(os.path.join(path_to_folder, args.camera))
        print(f"images: {len(images)}")
        img_obj_points = []
        img_img_points = []
        valid = False
        for idx, img_path in enumerate(images):
            img = cv2.imread(img_path)
            if img is None:
                continue
            print(f"process... idx {idx}")
            tag_results = detector.detect(img)
            print(f"tag_results ... {tag_results}")
            if len(tag_results) != 0:
                for tag_family, tags in tag_results.items():
                    for tag in tags:
                        img_img_points.append(tag.corners[:, 0, :].astype(np.float32))
                        img_obj_points.append(
                            CALIB_BOARD_PARAMS[tag_family]['corners'].reshape(-1, 4, 3)[tag.tag_id].astype(np.float32))
                        for idx, corner in enumerate(tag.corners):
                            cv2.circle(img, tuple(*corner.astype(int)), 12, (255, 0, 0), 6)
                if idx > 30:
                    print(f"processed .... {len(img_img_points), len(img_obj_points)}")
                    break
                valid = True
            elif not valid:
                break
        print(f"img_obj_points {len(img_obj_points)}, img_img_points {len(img_img_points)}")
        obj_points.extend(img_obj_points)
        img_points.extend(img_img_points)
    cv2.destroyAllWindows()
    obj_points = np.asarray(obj_points)
    img_points = np.asarray(img_points)
    print(f"calculate parameters... {len(obj_points), len(img_points)}")
    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, init_intrinsic, None, flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    print("final Camera Matrix : \n", cameraMatrix)
    print("final Camera Distort coeff : \n", distCoeffs)
    cx = float(cameraMatrix[0, 2])
    cy = float(cameraMatrix[1, 2])
    fx = float(cameraMatrix[0, 0])
    fy = float(cameraMatrix[1, 1])

    camera_param = {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "distCoeffs": distCoeffs.tolist(), "cameraMatrix": cameraMatrix.tolist()}
    # write resule to file
    yaml_file = os.path.join(args.root, args.camera + ".yaml")
    with open(yaml_file, 'w') as file:
        documents = yaml.dump(camera_param, file)
        print(f"Camera calibration result is saved to {yaml_file}")

    print("test undistort img...")
    cv2.namedWindow('undistorted', cv2.WINDOW_GUI_NORMAL)
    # # show undistorted img
    # for img in images:
    #     dst = cv2.undistort(img, cameraMatrix, distCoeffs)
    #     cv2.imshow('undistorted', dst)
    #     key = cv2.waitKey(0)
    #     if key == ord('q'):
    #         break


if __name__ == '__main__':
    main(parse_args())
