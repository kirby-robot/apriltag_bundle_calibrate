import numpy as np
import yaml
import cv2

class Camera:
    def __init__(self, camera_param_file) -> None:
        # read camera parameters
        with open(camera_param_file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
            self.cx = data["cx"]
            self.cy = data["cy"]
            self.fx = data["fx"]
            self.fy = data["fy"]
            self.distCoeffs = np.array(data["distCoeffs"]).flatten()
            if self.distCoeffs.shape[0] > 4:
                self.k1 = self.distCoeffs[0]
                self.k2 = self.distCoeffs[1]
                self.p1 = self.distCoeffs[2]
                self.p2 = self.distCoeffs[3]

            self.cameraMatrix = np.array(
                [self.fx, 0, self.cx, 0, self.fy, self.cy, 0, 0, 1]).reshape(3, 3)
        print(f"Camera parameters are loaded from {camera_param_file}"
              f"\nfx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, distCoeffs={self.distCoeffs}")

class CameraCV:
    def __init__(self, camera_param_file) -> None:
        print(f"camera_param_file .... {camera_param_file}")
        cv_file = cv2.FileStorage(camera_param_file, cv2.FILE_STORAGE_READ)
        self.cameraMatrix = cv_file.getNode('CameraMatrix').mat()
        self.cx = self.cameraMatrix[0, 2]
        self.cy = self.cameraMatrix[1, 2]
        self.fx = self.cameraMatrix[0, 0]
        self.fy = self.cameraMatrix[1, 1]
        self.distCoeffs = cv_file.getNode('D').mat()
        if self.distCoeffs.shape[0] > 4:
            self.k1 = self.distCoeffs[0]
            self.k2 = self.distCoeffs[1]
            self.p1 = self.distCoeffs[2]
            self.p2 = self.distCoeffs[3]

        print(f"Camera parameters are loaded from {camera_param_file}"
              f"\nfx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}, distCoeffs={self.distCoeffs}")
