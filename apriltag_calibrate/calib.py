import abc
from collections import OrderedDict
import os
import cv2
import numpy as np
import scipy.interpolate
from numpy.linalg import inv
import six
import yaml
from scipy import interpolate
from scipy.spatial.transform import Rotation

YAML_HEADER = "%YAML:1.0\n---\n"


def opencv_matrix_representer(dumper, mat):
    mat = np.atleast_2d(mat)
    mapping = OrderedDict([
        ('rows', mat.shape[0]),
        ('cols', mat.shape[1]),
        ('dt', 'd'),
        ('data', mat.reshape(-1).tolist())])
    return dumper.represent_mapping(u"tag:yaml.org,2002:opencv-matrix", mapping)


yaml.add_representer(OrderedDict, yaml.representer.SafeRepresenter.represent_dict)
yaml.add_representer(np.ndarray, opencv_matrix_representer)


class CalibrationReader(object):
    def __init__(self, calib_path):
        self.fs = cv2.FileStorage()
        ok = self.fs.open(calib_path, cv2.FILE_STORAGE_READ)
        if not ok:
            raise RuntimeError("Could not open '%s' (does it exist and is readable?) " % (calib_path))

    def __enter__(self):
        return self.fs

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fs.release()


@six.add_metaclass(abc.ABCMeta)
class Calibration(object):
    '''Define a calibration loader class for every type of sensors defined in: sensor.SensorType

    The required initialization parameters are specified in:
        - STR_KEYS (type: str)
        - INT_KEYS (type: int)
        - MAT_KEYS (type: np.array)
    '''

    STR_KEYS = ['date', 'type', 'sensor_name']
    INT_KEYS = []
    MAT_KEYS = []
    FLOAT_KEYS = []

    def __init__(self, **kwargs):
        for key in self.STR_KEYS + self.INT_KEYS + self.MAT_KEYS + self.FLOAT_KEYS:
            if key not in kwargs:
                raise RuntimeError(
                    "Missing required arg '%s' to initialize '%s'" % (key, type(self).__name__))
            setattr(self, key, kwargs[key])
        self.reset_params()

    @abc.abstractmethod
    def reset_params(self):
        pass

    def __eq__(self, other):
        '''WARNING: only compare the required initialization parameters.
        '''
        are_equal = True
        if not isinstance(other, type(self)):
            are_equal = False
        else:
            for key in self.STR_KEYS + self.INT_KEYS:
                are_equal = (getattr(self, key) == getattr(other, key))
                if not are_equal:
                    break
            for key in self.MAT_KEYS:
                are_equal = np.allclose(getattr(self, key), getattr(other, key), atol=1e-7)
                if not are_equal:
                    break
        return are_equal

    @staticmethod
    def load(calib_path):
        kwargs = {}
        with CalibrationReader(calib_path) as fs:
            # Find the appropriate calibration loader
            sensor_type = fs.getNode('type').string()
            if sensor_type.find("_") != -1:
                calib_cls_name = sensor_type.title().replace('_', '')
            else:
                calib_cls_name = sensor_type
                if calib_cls_name[0].islower():
                    calib_cls_name = calib_cls_name.title()
            # calib_cls = utils.import_class(calib_cls_name, __name__)
            calib_cls = globals()[calib_cls_name]
            # Load the required initialization parameters
            for str_key in calib_cls.STR_KEYS:
                kwargs[str_key] = fs.getNode(str_key).string()
            for int_key in calib_cls.INT_KEYS:
                kwargs[int_key] = int(fs.getNode(int_key).real())
            for mat_key in calib_cls.MAT_KEYS:
                kwargs[mat_key] = np.array(fs.getNode(mat_key).mat(), dtype=np.float32)
            for float_key in calib_cls.FLOAT_KEYS:
                kwargs[float_key] = float(fs.getNode(float_key).real())
            kwargs['calib_path'] = calib_path
        return calib_cls(**kwargs)

    def save(self, calib_path):
        fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_WRITE)
        self.__setattr__("type", self.__class__.__name__)
        for str_key in self.__class__.STR_KEYS:
            fs.write(str_key, self.__getattribute__(str_key))
        for int_key in self.__class__.INT_KEYS:
            fs.write(int_key, self.__getattribute__(int_key))
        for mat_key in self.__class__.MAT_KEYS:
            fs.write(mat_key, self.__getattribute__(mat_key))
        for float_key in self.__class__.FLOAT_KEYS:
            fs.write(float_key, self.__getattribute__(float_key))

    def to_dicts(self, ):
        kwargs = {"type": self.__class__.__name__}
        for str_key in self.__class__.STR_KEYS:
            kwargs[str_key] = self.__getattribute__(str_key)
        for int_key in self.__class__.INT_KEYS:
            kwargs[int_key] = self.__getattribute__(int_key)
        for mat_key in self.__class__.MAT_KEYS:
            kwargs[mat_key] = self.__getattribute__(mat_key).copy()
        for float_key in self.__class__.FLOAT_KEYS:
            kwargs[float_key] = self.__getattribute__(float_key)
        kwargs['type'] = self.__class__.__name__
        return kwargs


@six.add_metaclass(abc.ABCMeta)
class CameraCalibration(Calibration):
    INT_KEYS = ['height', 'width', 'rotate']
    MAT_KEYS = ['R', 'P', 'M', 'D', 'Tr_cam_to_imu']

    def reset_params(self):
        self.imgsize = (self.width, self.height)
        self.init_undistort_map()
        self.Tr_imu_to_cam = inv(self.Tr_cam_to_imu)

    @abc.abstractmethod
    def init_undistort_map(self):
        pass

    @abc.abstractmethod
    # From: https://github.com/egonSchiele/OpenCV/blob/master/modules/imgproc/src/undistort.cpp#L193
    def undistort(self, img):
        """
        Default cv2 undistort with unity Rotation matrix
        """
        pass

    def undistort_rectify(self, img):
        """
        Instead of using cv2.undistort, which is basically combination of
        initUndistortRectifyMap() (with unity R) and remap() (with bilinear interpolation);
        we use the rotation matrix in calib file to do initUndistortRectifyMap() in
        self.init_undistort_map() and then use cv2.remap using this function.

        Refer:
        https://docs.opencv.org/2.4.13.7/modules/imgproc/doc/geometric_transformations.html#undistort
        """
        resized_img = cv2.resize(img, self.imgsize)
        return cv2.remap(resized_img, self.map1, self.map2, cv2.INTER_LINEAR)

    @abc.abstractmethod
    def projectPoints(self, points):
        """

        :param points:
        :return:
        """
        pass

    def resize(self, ratios):
        """
        return a new calib after resize.
        :param ratios:
        :return:
        """
        raise NotImplementedError(f"{self.__class__.__name__} not support resize!")

    def crop(self, top_x, top_y):
        """
        return a new calib after crop image.
        :param top_x:
        :param top_y:
        :return:
        """

        raise NotImplementedError(f"{self.__class__.__name__} not support Crop!")

    def padding(self, top_x, top_y):
        """
        return a new calib after padding camera.
        :param top_x:`
        :param top_y:
        :return:
        """

        raise NotImplementedError(f"{self.__class__.__name__} not support Crop!")

    def hflip(self):
        """

        :return:
        """
        raise NotImplementedError(f"{self.__class__.__name__} not support hflip!")

    def vflip(self):
        """

        :return:
        """
        raise NotImplementedError(f"{self.__class__.__name__} not support vflip!")


class Camera(CameraCalibration):

    def projectPoints(self, points):
        # print(f"M {self.M}, D {self.D}")
        pts_uvs, _ = cv2.projectPoints(points, np.array([0., 0., 0], dtype=np.float64),
                                       np.array([0, 0., 0.], dtype=np.float64),
                                       self.M.astype(np.float64), self.D.reshape(-1).astype(np.float64))
        pts_uvs = pts_uvs.reshape((points.shape[0], 2))
        return pts_uvs

    def undistort(self, img):
        resized_img = cv2.resize(img, self.imgsize)
        undist_img = cv2.undistort(resized_img, self.M, self.D, newCameraMatrix=self.P)
        return undist_img

    def init_undistort_map(self):
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.M, self.D, self.R, self.P, self.imgsize, cv2.CV_32FC1)

    def undistort_points(self, points, R=None, use_m=True):
        points = np.array(points).reshape((-1, 2))
        if R is None:
            R = np.eye(3)
        P = self.P
        if use_m:
            P = self.M
        # TODO: if for camera
        undistort_pts = cv2.undistortPoints(points, self.M, self.D,
                                            R=R, P=P)
        return undistort_pts.reshape(-1, 2)

    def image_2_ground(self, imu_height, image_points, distort=True):
        """

        :param imu_height:
        :param image_points:
        :param distort: True or False, True means do distortion for given image points.
        :return:
        """
        image_points = np.array(image_points).reshape((-1, 2))
        if distort:
            undistort_points = self.undistort_points(image_points, R=np.eye(3), use_m=True)
        else:
            undistort_points = image_points
        ones = np.ones(undistort_points.shape[0], dtype=np.float32).reshape((-1, 1))
        undistort_pts_homo = np.concatenate([undistort_points, ones], axis=-1).T
        M_inv = np.linalg.inv(self.M)
        pts_3d = (self.Tr_cam_to_imu[:3, :3] @ (M_inv @ undistort_pts_homo)).T

        z = -imu_height * np.ones(pts_3d.shape[0], dtype=np.float32)
        scale = (z - self.Tr_cam_to_imu[2, 3]) / pts_3d[:, 2]
        x = self.Tr_cam_to_imu[0, 3] + pts_3d[:, 0] * scale
        y = self.Tr_cam_to_imu[1, 3] + pts_3d[:, 1] * scale

        return np.vstack([x, y, z]).transpose().reshape((*image_points.shape[:-1], 3))

    def __str__(self):
        return f"Camera: width, height: {self.width}, {self.height}, M: {self.M}," \
               f" Tr_cam_to_imu: {self.Tr_cam_to_imu}, D: {self.D}"

    def resize(self, ratios):
        """

        :param ratios:
        :return:
        """
        if isinstance(ratios, (float, int)):
            ratios = np.array([ratios, ratios], dtype=np.float32)
        elif isinstance(ratios, (list, tuple)):
            ratios = np.array(ratios, dtype=np.float32)

        assert len(ratios) == 2

        height = int(round(self.height * ratios[1]))
        width = int(round(self.width * ratios[0]))
        M = self.M.copy()
        M[:2, :] = M[:2, :] * ratios.reshape(2, 1)

        kwargs = self.to_dicts()
        kwargs['height'] = height
        kwargs['width'] = width
        kwargs['M'] = M
        return Camera(**kwargs)

    def crop(self, tl_x: int, tl_y: int):
        """

        :param tl_x:
        :param tl_y:
        :return:
        """
        height = self.height - tl_y
        width = self.width - tl_x
        M = self.M.copy()
        M[0, 2] -= tl_x
        M[1, 2] -= tl_y
        kwargs = self.to_dicts()
        kwargs['height'] = height
        kwargs['width'] = width
        kwargs['M'] = M
        return Camera(**kwargs)

    def padding(self, tl_x, tl_y):
        """

        :param tl_x:
        :param tl_y:
        :return:
        """
        height = self.height + tl_y
        width = self.width + tl_x
        M = self.M.copy()
        M[0, 2] += tl_x
        M[1, 2] += tl_y
        kwargs = self.to_dicts()
        kwargs['height'] = height
        kwargs['width'] = width
        kwargs['M'] = M
        return Camera(**kwargs)

    def hflip(self, ):
        """

        :return:
        """
        M = self.M.copy()
        M[0, 2] = self.width - M[0, 2]
        M[0, 0] = - M[0, 0]
        kwargs = self.to_dicts()
        kwargs['M'] = M
        return Camera(**kwargs)

    def vflip(self):
        """

        :return:
        """
        M = self.M.copy()
        M[1, 2] = self.height - M[1, 2]
        M[1, 1] = - M[1, 1]
        kwargs = self.to_dicts()
        kwargs['M'] = M
        return Camera(**kwargs)


class StereoCamera(Calibration):
    '''Reminder:
          [R T] = Tr_ref_cam_to_sec_cam
          R1 = Tr_ref_cam_to_rectify_cam
          R2 = Tr_sec_cam_to_rectify_cam
          R = inv(R2)*R1
    '''
    INT_KEYS = ['height', 'width', 'rotate']
    MAT_KEYS = ['R', 'T', 'Q', 'R1', 'P1', 'M1', 'D1', 'R2', 'P2', 'M2', 'D2', 'Tr_cam_to_imu']

    def reset_params(self):
        self.imgsize = (self.width, self.height)
        self.Q_inv = inv(self.Q)
        self.Tr_imu_to_cam = inv(self.Tr_cam_to_imu)
        # Init the corresponding mono cameras
        camera_params = {
            key: getattr(self, key) for key in self.STR_KEYS + self.INT_KEYS + ['Tr_cam_to_imu']
        }
        camera_params.update({
            'R': self.R1, 'P': self.P1, 'M': self.M1, 'D': self.D1,
        })
        self.left_camera = Camera(**camera_params)
        camera_params.update({
            'R': self.R2, 'P': self.P2, 'M': self.M2, 'D': self.D2,
        })
        self.right_camera = Camera(**camera_params)

    def as_mono_cameras(self):
        return self.left_camera, self.right_camera

    def undistort_left(self, left_img):
        return self.left_camera.undistort(left_img)

    def undistort_right(self, right_img):
        return self.right_camera.undistort(right_img)

    def undistort_rectify_left(self, left_img):
        return self.left_camera.undistort_rectify(left_img)

    def undistort_rectify_right(self, right_img):
        return self.right_camera.undistort_rectify(right_img)


class OmiCamera(CameraCalibration):
    """
    reference :https://github.com/valgur/ocam/blob/master/ocam/reprojectpoints.py
    """
    INT_KEYS = ['height', 'width', 'rotate']
    MAT_KEYS = ['R', 'Tr_cam_to_imu', 'stretch_matrix', 'pol']
    FLOAT_KEYS = ['a0', 'a2', 'a3', 'a4', 'cx', 'cy']

    def reset_params(self):
        radius = max(self.width - self.cx, self.cx)
        pol, err, _ = self.findInvPoly(radius)
        self.pol = pol

    def undistort(self, img):
        return img

    def undistort_rectify(self, img):
        return img

    def init_undistort_map(self):
        pass

    def findInvPoly(self, radius, N=None, tol=0.01):
        """
        reference: https://github.com/valgur/ocam/blob/master/ocam/findinvpoly.py
        """
        theta = np.arange(-np.pi / 2.0, np.pi / 2.0, 0.01)

        def find_lambda(coeff):
            m = np.tan(theta)
            r = np.zeros(len(m))
            poly_coef = coeff[::-1]
            poly_coef_tmp = poly_coef.copy()
            for i in range(len(m)):
                poly_coef_tmp[-2] = poly_coef[-2] - m[i]
                rho_tmp = cv2.solvePoly(poly_coef_tmp[::-1], maxIters=50)[1][:, 0]
                mask = (rho_tmp[:, 0] > 0) & (np.abs(rho_tmp[:, 1]) < 1e-10)
                res = rho_tmp[mask, 0]
                r[i] = np.min(res) if len(res) > 0 else np.inf
            return r

        def fit_poly_and_estimate_error(roots, order):
            pol = np.polyfit(theta, roots, order)
            err = np.abs(roots - np.polyval(pol, theta))
            return pol, err, order

        coef = np.array([self.a0, 0, self.a2, self.a3, self.a4], dtype=np.float32)
        r_v = find_lambda(coef)
        valid = r_v < radius
        theta = theta[valid]
        r_v = r_v[valid]
        if N is None:
            maxerr = np.inf
            N = 1
            while maxerr > tol:
                N += 1
                pol, err, _ = fit_poly_and_estimate_error(r_v, N)
                maxerr = np.max(err)
        else:
            pol, err, N = fit_poly_and_estimate_error(r_v, N)
        return pol, err, N

    def projectPoints(self, points):
        """

        :param points:
        :return:
        """
        norm = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        norm[norm == 0] = np.finfo(float).eps
        v = points / norm.reshape(-1, 1)
        theta = np.arctan(v[:, 2])
        rho = np.polyval(self.pol, theta)
        v[:, 0] = v[:, 0] * rho
        v[:, 1] = v[:, 1] * rho
        uv = (self.stretch_matrix @ v[:, :2].transpose()).transpose()
        uv[:, 0] += self.cx
        uv[:, 1] += self.cy

        return uv

    def resize(self, ratio):
        """

        :param ratio: float or tuples
        :return:
        """
        if isinstance(ratio, (float, int)):
            ratio = np.array([[ratio, 0], [0, ratio]], dtype=np.float32)
        elif isinstance(ratio, (list, tuple, np.ndarray)):
            ratio = np.array([[ratio[0], 0], [0, ratio[1]]], dtype=np.float32)
        kwargs = self.to_dicts()
        stretch_matrix = ratio @ kwargs['stretch_matrix']
        kwargs['stretch_matrix'] = stretch_matrix
        kwargs['width'] = int(round(self.width * ratio[0, 0]))
        kwargs['height'] = int(round(self.height * ratio[1, 1]))
        kwargs['cx'] = self.cx * ratio[0, 0]
        kwargs['cy'] = self.cy * ratio[1, 1]
        return self.__class__(**kwargs)

    def crop(self, top_x: int, top_y: int):
        """

        :param top_x:
        :param top_y:
        :return:
        """
        height = self.height - top_y
        width = self.width - top_x
        cx = self.cx - top_x
        cy = self.cy - top_y

        kwargs = self.to_dicts()
        kwargs['height'] = height
        kwargs['width'] = width
        kwargs['cx'] = cx
        kwargs['cy'] = cy
        return OmiCamera(**kwargs)

    def padding(self, top_x, top_y):
        """

        :param top_x:
        :param top_y:
        :return:
        """
        height = self.height + top_y
        width = self.width + top_x
        cx = self.cx + top_x
        cy = self.cy + top_y

        kwargs = self.to_dicts()
        kwargs['height'] = height
        kwargs['width'] = width
        kwargs['cx'] = cx
        kwargs['cy'] = cy
        return OmiCamera(**kwargs)

    def hflip(self):
        """

        :return:
        """
        cx = self.width - self.cx
        kwargs = self.to_dicts()
        # means flip the first column of stretch matrix.
        kwargs['stretch_matrix'] = np.array([[-1, 0], [0, 1]], dtype=np.float) @ self.stretch_matrix
        kwargs['cx'] = cx
        return OmiCamera(**kwargs)

    def vflip(self):
        cy = self.height - self.cy
        kwargs = self.to_dicts()
        kwargs['stretch_matrix'] = np.array([[1, 0], [0, -1]], dtype=np.float) @ self.stretch_matrix
        kwargs['cy'] = cy
        return OmiCamera(**kwargs)

    def image_2_ground(self, imu_height, point_uvs, scale=1.0):
        """
        compute the 3d point by ground point
        :param imu_height: float
        :param point_uvs: np.array() N * 2
        :param scale: for resized image, should set the scale parameter
        :return:
        """
        cx = self.cx / scale
        cy = self.cy / scale
        a0 = self.a0 / scale
        a2 = self.a2 * scale
        a3 = self.a3 * (scale ** 2)
        a4 = self.a4 * (scale ** 3)
        uv = np.matmul(np.linalg.inv(self.stretch_matrix), (point_uvs - np.array([cx, cy])).transpose())
        rho = np.sqrt(np.sum(uv ** 2, axis=0))
        rho = a0 + a2 * (rho ** 2) + a3 * (rho ** 3) + a4 * (rho ** 4)
        pts_3d = np.concatenate([uv, rho.reshape(1, -1)], axis=0)
        # print(f"fisheye image_2_ground {pts_3d}")
        pts_3d = np.matmul(self.Tr_cam_to_imu[: 3, :3], pts_3d)
        # print(f"after to imu, {pts_3d}")
        pts_3d = pts_3d[:, :] / pts_3d[2, :]
        z = -imu_height
        x = self.Tr_cam_to_imu[0, 3] + pts_3d[0, :] * (z - self.Tr_cam_to_imu[2, 3]) / pts_3d[2, :]
        y = self.Tr_cam_to_imu[1, 3] + pts_3d[1, :] * (z - self.Tr_cam_to_imu[2, 3]) / pts_3d[2, :]
        return np.concatenate([x.reshape(1, -1), y.reshape(1, -1), z * np.ones_like(y).reshape(1, -1)],
                              axis=0).transpose()


class CylinderCamera(Camera):
    """
    TODO: refactor this class.
    """
    INT_KEYS = ['height', 'width']
    MAT_KEYS = ['Tr_cam_to_imu', 'R', 'M']
    FLOAT_KEYS = ['cx', 'cy', 'fx', 'fy']

    def undistort(self, img):
        pass

    def reset_params(self):
        pass

    @staticmethod
    def compute_intrinsics(horizontal_fov, vertical_fov, image_width, image_height):
        """
        :param horizontal_fov:
        :param vertical_fov:
        :param image_width:
        :param image_height:
        :return:
        """
        # suppose f_phi = f_theta, and assume always make f_theta = image_height / 2 / np.tan(vertical_fov/2.0)
        fx = image_width / horizontal_fov
        fy = fx
        # then vertical fov: math.atan(image_height / 2.0 / fy) * 2.
        # actually for raw image is about 106.26 degree.
        cx = image_width / 2.0
        # based on the vertical_fov should be covered.
        cy = int(image_height - fy * np.tan(vertical_fov))
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float32)
        return intrinsics, image_height

    @staticmethod
    def build_from_fisheye(fisheye_calib, image_height, image_width, rectify_matrix, horizontal_fov, vertical_fov):
        intrinsic, valid_height = CylinderCamera.compute_intrinsics(horizontal_fov, vertical_fov, image_width,
                                                                    image_height)
        print(f"intrinsics ... {intrinsic}, valid_height {valid_height}, cylinder_height {image_height}")

        Tr_imu_to_cylinder = np.zeros((4, 4), dtype=np.float32)
        Tr_imu_to_cylinder[:3, :3] = rectify_matrix
        Tr_imu_to_cylinder[3, 3] = 1.0
        # T_cam_to_imu represents the translation from camera to imu in imu frame.
        # And T_imu_to_cam represents the translation from imu to camera in camera frame.
        # so it equals to rectify_matrix * (imu_to_camera) = - rectify_matrix * T_cam_to_imu.
        Tr_imu_to_cylinder[:3, 3] = -np.matmul(rectify_matrix, fisheye_calib.Tr_cam_to_imu[:3, 3]).reshape(-1)
        kwargs = {
            "height": image_height,
            "width": image_width,
            "Tr_cam_to_imu": np.linalg.inv(Tr_imu_to_cylinder),
            "R": rectify_matrix,
            "M": intrinsic,
            "type": "cylinder",
            "cx": intrinsic[0, 2],
            "cy": intrinsic[1, 2],
            "fx": intrinsic[0, 0],
            "fy": intrinsic[1, 1]
        }
        for key in Calibration.STR_KEYS:
            kwargs[key] = getattr(fisheye_calib, key)
        kwargs['type'] = 'cylinder_camera'
        cylinder_calib = CylinderCamera(**kwargs)
        return cylinder_calib

    def save_to_yaml(self, calib_file):
        fs = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_WRITE)
        for key in self.STR_KEYS + self.INT_KEYS + self.FLOAT_KEYS + self.MAT_KEYS:
            fs.write(key, self.__getattribute__(key))
        fs.release()

    @staticmethod
    def ground_depth(point_uvs, intrinsic, Tr_cam_to_imu, imu_height):
        y = imu_height + Tr_cam_to_imu[2, 3]
        rho = y * intrinsic[1, 1] / (point_uvs[:, 1] - intrinsic[1, 2] + 1e-6)
        theta = (point_uvs[:, 0] - intrinsic[0, 2]) / intrinsic[0, 0]
        x = rho * np.sin(theta)
        z = rho * np.cos(theta)
        pts_cam = np.concatenate([x.reshape(1, -1), y * np.ones_like(x).reshape(1, -1), z.reshape(1, -1)], axis=0)
        # N * 3
        return pts_cam.transpose()

    def image_2_ground(self, imu_height, point_uvs, **kwargs):
        """
        compute the 3d point by ground point
        :param imu_height: float
        :param point_uvs: np.array() N * 2
        :return:
        """
        pts_cam = self.ground_depth(point_uvs, self.M, self.Tr_cam_to_imu, imu_height).transpose()
        pts_cam = np.concatenate([pts_cam, np.ones(pts_cam.shape[1]).reshape(1, -1)], axis=0)
        pts_imu = np.matmul(self.Tr_cam_to_imu, pts_cam).transpose()
        print(f"cylinder pts_imu {pts_imu}")
        return pts_imu[:, :3]

    def projectPoints(self, points):
        theta = np.arctan2(points[0, :], points[2, :])
        rho = np.sqrt(points[0, :] ** 2 + points[2, :] ** 2)
        y = points[1, :] / rho
        pts_3d_cylinder = np.concatenate([theta.reshape(-1, 1),
                                          y.reshape(-1, 1),
                                          np.ones_like(y).reshape(-1, 1)], axis=1)
        project_pts = self.M @ pts_3d_cylinder.transpose()
        return project_pts[:2, :].transpose()


class FisheyeTableCamera(CameraCalibration):
    """
    Fisheye camera with provided distort table.
    """
    INT_KEYS = ['height', 'width', 'rotate']
    MAT_KEYS = ['R', 'Tr_cam_to_imu']
    FLOAT_KEYS = ['a0', 'cx', 'cy', 'fov_ratio', 'pixel_size']
    STR_KEYS = ['car', 'date', 'type', 'sensor_name', 'distort_table_file', 'calib_path']

    def init_undistort_map(self):
        """
        project points from undistort image to distort image.
        :return:
        """
        map_x = - np.ones((self.height, self.width), dtype=np.float32)
        map_y = - np.ones((self.height, self.width), dtype=np.float32)

        x = np.arange(self.width, dtype=np.int32)
        y = np.arange(self.height, dtype=np.int32)
        shift_x, shift_y = np.meshgrid(x, y)
        x_shift = np.expand_dims(shift_x, axis=-1)
        y_shift = np.expand_dims(shift_y, axis=-1)
        # X, Y, 2
        coords = np.concatenate([x_shift, y_shift], axis=-1)
        uvs_cam = coords.reshape(-1, 2).astype(np.float32)

        center = np.array([self.cx, self.cy], dtype=np.float32)
        points_undistort = (uvs_cam - center) * self.fov_ratio * self.pixel_size
        ru = np.linalg.norm(points_undistort, axis=-1)
        rd = self.f_ru_to_rd(ru)
        points_cam = np.divide(rd, ru)[..., None] * points_undistort
        points_cam = points_cam / self.pixel_size + center
        points_cam = points_cam.reshape((self.height, self.width, 2))
        for v in range(self.height):
            for u in range(self.width):
                if 0 <= points_cam[v, u, 0] < self.width and 0 <= points_cam[v, u, 1] < self.height:
                    map_x[v, u] = points_cam[v, u, 0]
                    map_y[v, u] = points_cam[v, u, 1]

        self.map_x = map_x
        self.map_y = map_y

    def projectPoints(self, points):
        """
        :param points: 3d points in camera coorindates.
        :return:
        """
        uvs_cam, _ = cv2.projectPoints(points, rvec=np.eye(3), tvec=np.zeros((3, 1)), cameraMatrix=self.M,
                                       distCoeffs=self.D)
        uvs_cam = uvs_cam.reshape(-1, 2)
        center = np.array([self.cx, self.cy])
        uvs_undistort = (uvs_cam - center) * self.fov_ratio * self.pixel_size
        ru = np.linalg.norm(uvs_undistort, axis=-1)
        rd = self.f_ru_to_rd(ru)
        points_cam = np.divide(rd, ru)[..., None] * uvs_undistort
        points_cam = points_cam / self.pixel_size + center
        return points_cam

    def undistort(self, img):
        return self.undistort_rectify(img)

    def undistort_rectify(self, img):
        resized_img = cv2.resize(img, self.imgsize)
        return cv2.remap(resized_img, self.map1, self.map2, cv2.INTER_LINEAR)

    def undistort_points(self, points):
        points = np.array(points).reshape((-1, 2))

        center = np.array([self.cx, self.cy], dtype=np.float32)
        points_cam = (points - center) * self.pixel_size
        rd = np.linalg.norm(points_cam, axis=-1)
        ru = self.f_rd_to_ru(rd)
        uvs_undistort = np.divide(ru, rd)[..., None] * points_cam
        uvs_undistort = uvs_undistort / self.pixel_size / self.fov_ratio + center

        uvs_undistort = uvs_undistort.reshape(-1, 2)
        return uvs_undistort

    def reset_params(self):
        """
        load distort_table_file
        :return:
        """
        distort_table_file = os.path.join(os.path.dirname(self.calib_path), self.distort_table_file)
        print(f"FisheyeTableCamera distort_table_file {distort_table_file}")
        # load distort_table_file
        distort_map = np.loadtxt(open(distort_table_file, encoding='utf-8'), delimiter=',')
        rds = distort_map[:, 1]
        rus = distort_map[:, 2]
        self.f_ru_to_rd = interpolate.interp1d(rus, rds, kind='cubic', fill_value="extrapolate")
        self.f_rd_to_ru = interpolate.interp1d(rds, rus, kind="cubic", fill_value="extrapolate")
        # virtual intrinsic.
        self.M = np.array([[self.a0 / self.fov_ratio, 0, self.width * 0.5],
                           [0, self.a0 / self.fov_ratio, self.height * 0.5],
                           [0, 0, 1]], dtype=np.float32)
        self.D = np.zeros((1, 5))

    def image_2_ground(self, imu_height, point_uvs):
        """

        compute the 3d point by ground point
        :param imu_height: float
        :param point_uvs: np.array() N * 2
        :return:
        """
        center = np.array([self.cx, self.cy], dtype=np.float32)
        points_cam = (point_uvs - center) * self.pixel_size
        rd = np.linalg.norm(points_cam, axis=-1)
        ru = self.f_rd_to_ru(rd)
        uvs_undistort = np.divide(ru, rd)[..., None] * points_cam
        uvs_undistort = uvs_undistort / self.pixel_size / self.fov_ratio + center

        uvs_undistort = uvs_undistort.reshape(-1, 2)

        ones = np.ones(uvs_undistort.shape[0], dtype=np.float32).reshape((-1, 1))
        undistort_pts_homo = np.concatenate([uvs_undistort, ones], axis=1)
        M_inv = np.linalg.inv(self.M)
        pts_3d = self.Tr_cam_to_imu[:3, :3] @ (M_inv @ undistort_pts_homo.transpose())
        pts_3d = pts_3d.transpose()
        z = -imu_height * np.ones(pts_3d.shape[0], dtype=np.float32)
        scale = (z - self.Tr_cam_to_imu[2, 3]) / pts_3d[:, 2]
        x = self.Tr_cam_to_imu[0, 3] + pts_3d[:, 0] * scale
        y = self.Tr_cam_to_imu[1, 3] + pts_3d[:, 1] * scale
        return np.vstack([x, y, z]).transpose()


class EquiCamera(CameraCalibration):
    FLOAT_KEYS = ['pitch', 'roll', 'yaw', 'distort', 'cam_height']
    INT_KEYS = ['height', 'width']
    MAT_KEYS = []

    def reset_params(self):
        self.Tr_cam_to_imu = np.array([[1, 0, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, 0, 1]], dtype=np.float32)

    def init_undistort_map(self):
        pass

    def undistort(self, img):
        return img

    def undistort_rectify(self, img):
        return img

    def projectPoints(self, points):
        pts = points / np.linalg.norm(points, axis=-1)[..., None]

        x, y, z = np.split(pts, 3, axis=-1)
        u = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        v = np.arctan2(y, c)

        u = (u + np.pi) * self.width / (2 * np.pi)
        v = (v + np.pi) * self.height / np.pi
        return np.concatenate([u, v], axis=-1)

    def distort_xyz_imu(self, xyz, x_range):
        xyz = xyz / xyz[..., 2][..., None] * (-self.cam_height)
        dy = np.abs(xyz[..., 0]) / x_range
        xyz[..., 1] *= (1.0 + dy * self.distort)
        return xyz

    def undistort_xyz_imu(self, xyz, x_range):
        xyz = xyz / xyz[..., 2][..., None] * (-self.cam_height)
        dy = np.abs(xyz[..., 0]) / x_range
        xyz[..., 1] /= (1.0 + dy * self.distort)
        return xyz

    def rotate_xyz(self, xyz):
        r = Rotation.from_rotvec([self.roll, self.pitch, self.yaw]).as_matrix()
        return (r @ xyz.reshape(-1, 3).T).T.reshape(xyz.shape)

    def update_params(self, d_pitch=0, d_roll=0, d_yaw=0, d_distort=0):
        self.pitch += d_pitch
        self.roll += d_roll
        self.yaw += d_yaw
        self.distort += d_distort

    def image_2_ground(self, imu_height, pts):
        """

        :param pts:
        :return:
        """


class BaiduPinholeCamera(Camera):
    """
    """
    STR_KEYS = []
    FLOAT_KEYS = ['pitch', 'yaw', 'fov', "imu_height"]
    INT_KEYS = ['width', 'height']
    MAT_KEYS = []

    def reset_params(self):
        cx = self.width * 0.5
        cy = self.height * 0.5
        f = (self.width * 0.5) / np.tan(self.fov * np.pi / 180 * 0.5)
        self.M = np.array([[f, 0, cx],
                           [0, f, cy],
                           [0, 0, 1]])
        self.P = self.M
        self.D = np.zeros((1, 5))
        self.Tr_cam_to_imu = np.array([[0, 0, 1, 0],
                                       [-1, 0, 0, 0],
                                       [0, -1, 0, 0],
                                       [0, 0, 0, 1]], dtype=np.float32)

    def init_undistort_map(self):
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        uv = np.concatenate([u[..., None], v[..., None]], axis=-1)
        return uv[..., 0], uv[..., 1]

    def projectPoints(self, points):
        uv, _ = cv2.projectPoints(points.reshape(-1, 3), rvec=np.eye(3), tvec=np.zeros((3, 1)),
                                  cameraMatrix=self.M,
                                  distcoeffs=self.D)
        uv = uv.reshape(*(points.shape[:-1]), 2)
        return uv


class SphereTableCamera(CameraCalibration):
    """
    Use Sphere with distort table to describe the Camera.
    """
    INT_KEYS = ['height', 'width']
    STR_KEYS = ['type', 'distort_table_file', 'calib_path']
    FLOAT_KEYS = ['a0', 'a2', 'a3', 'a4', 'cx', 'cy', 'fov_ratio', 'pixel_size']
    MAT_KEYS = ['Tr_cam_to_imu', 'stretch_matrix', 'M', 'R', 'P', 'D']

    def reset_params(self):
        distort_table_file = os.path.join(os.path.dirname(self.calib_path), self.distort_table_file)
        distort_map = np.loadtxt(open(distort_table_file, encoding='utf-8'), delimiter=",")
        theta_u = distort_map[:, 0]
        rd = distort_map[:, 1]
        ru = distort_map[:, 2]
        self.ru2rd = scipy.interpolate.interp1d(ru, rd, kind='cubic', fill_value="extrapolate")
        self.rd2ru = scipy.interpolate.interp1d(rd, ru, kind='cubic', fill_value="extrapolate")
        self.thetau2rd = scipy.interpolate.interp1d(theta_u, rd, kind='cubic', fill_value="extrapolate")
        self.M = np.array([[self.a0 / self.fov_ratio, 0, self.cx],
                           [0, self.a0 / self.fov_ratio, self.cy],
                           [0, 0, 1]], dtype=np.float32)
        self.D = np.zeros((1, 5))

    def init_undistort_map(self):
        u, v = np.meshgrid(np.arange(self.width * self.fov_ratio),
                           np.arange(self.height * self.fov_ratio))
        uv = np.concatenate([u[..., None], v[..., None]], axis=-1)
        uv = self.distort_points(uv)
        uv = cv2.resize(uv, (self.width, self.height)).astype(np.float32)
        return uv[..., 0], uv[..., 1]

    def undistort(self, img):
        pass

    def projectPoints(self, xyz):
        theat_u = np.arctan2(np.linalg.norm(xyz[..., :2], axis=-1), xyz[..., 2])
        # # theta_u -> rd
        rd = self.thetau2rd(theat_u * 180 / np.pi)
        v = rd / (np.sqrt(1 + np.power(xyz[..., 0] / xyz[..., 1], 2)) * self.pixel_size)
        v = v * xyz[..., 1] / np.abs(xyz[..., 1])
        u = (xyz[..., 0] / xyz[..., 1]) * v
        u = u + self.cx
        v = v + self.cy

        return np.concatenate([u[..., None], v[..., None]], axis=-1)

    def undistort_points(self, pts):
        """

        :param pts:
        :return:
        """
        pts[..., 0] = (pts[..., 0] - self.cx * self.fov_ratio) * self.pixel_size
        pts[..., 1] = (pts[..., 1] - self.cy * self.fov_ratio) * self.pixel_size
        eps = 1e-9
        ru = np.linalg.norm(pts, axis=-1) + eps
        rd = self.ru2rd(ru)

        pts[..., 0] = np.divide(ru, rd) * pts[..., 0]
        pts[..., 1] = np.divide(ru, rd) * pts[..., 1]

        # pts[..., 0] = np.divide(rd, ru) * pts[..., 0]
        # pts[..., 1] = np.divide(rd, ru) * pts[..., 1]

        pts[..., 0] = pts[..., 0] / self.pixel_size + self.cx
        pts[..., 1] = pts[..., 1] / self.pixel_size + self.cy

        return pts

    def distort_points(self, pts):
        '''
            pts: (..., 2)
        '''
        pts[..., 0] = (pts[..., 0] - self.cx * self.fov_ratio) * self.pixel_size
        pts[..., 1] = (pts[..., 1] - self.cy * self.fov_ratio) * self.pixel_size
        eps = 1e-9
        ru = np.linalg.norm(pts, axis=-1) + eps
        rd = self.ru2rd(ru)

        pts[..., 0] = np.divide(rd, ru) * pts[..., 0]
        pts[..., 1] = np.divide(rd, ru) * pts[..., 1]

        pts[..., 0] = pts[..., 0] / self.pixel_size + self.cx
        pts[..., 1] = pts[..., 1] / self.pixel_size + self.cy

        return pts

    def image_2_ground(self, imu_height, point_uvs, **kwargs):
        """

        :return:
        """

        uvs_undistort = self.undistort_points(point_uvs)

        ones = np.ones(uvs_undistort.shape[0], dtype=np.float32).reshape((-1, 1))
        undistort_pts_homo = np.concatenate([uvs_undistort, ones], axis=1)
        M_inv = np.linalg.inv(self.M)
        pts_3d = self.Tr_cam_to_imu[:3, :3] @ (M_inv @ undistort_pts_homo.transpose())
        pts_3d = pts_3d.transpose()
        z = -imu_height * np.ones(pts_3d.shape[0], dtype=np.float32)
        scale = (z - self.Tr_cam_to_imu[2, 3]) / pts_3d[:, 2]
        x = self.Tr_cam_to_imu[0, 3] + pts_3d[:, 0] * scale
        y = self.Tr_cam_to_imu[1, 3] + pts_3d[:, 1] * scale
        return np.vstack([x, y, z]).transpose()


class IpmCamera(CameraCalibration):
    """
    For the moment, we make it a dynamic generate calib.
    """
    STR_KEYS = ['type']
    INT_KEYS = ['width', 'height', 'rotate', 'imu_height']
    MAT_KEYS = ['Tr_cam_to_imu', 'M', 'D']

    def reset_params(self):
        pass

    def init_undistort_map(self):
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        uv = np.concatenate([u[..., None], v[..., None]], axis=-1)
        return uv[..., 0], uv[..., 1]

    def projectPoints(self, points, frame="camera"):
        """
        points in camera coordinate
        """
        assert frame in ['camera', 'imu']
        if frame == 'imu':
            pass
        uv, _ = cv2.projectPoints(points.reshape(-1, 3), rvec=np.eye(3), tvec=np.zeros(3),
                                  cameraMatrix=self.M, distCoeffs=self.D)
        uv = uv.reshape(*(points.shape[:-1]), 2)
        return uv

    def image_2_ground(self, imu_height, point_uvs, **kwargs):
        """

        """
        depth = np.abs(self.Tr_cam_to_imu[2, 3]) + imu_height
        shape = point_uvs.shape
        point_uvs = point_uvs.reshape((-1, 2))
        ones = np.ones((point_uvs.shape[0], 1))
        points_xyz = np.concatenate([point_uvs, ones], axis=-1)
        points_xyz = np.linalg.inv(self.M) @ points_xyz.T * depth
        xyz_imu = (self.Tr_cam_to_imu[:3, :3] @ points_xyz +
                   self.Tr_cam_to_imu[:3, 3:]).T.reshape((*shape[:-1], 3))
        return xyz_imu

    def undistort(self, img):
        pass

    def __str__(self):
        return f"width {self.width}, height {self.height}, M {self.M}, Tr_cam_to_imu {self.Tr_cam_to_imu}"

    def get_undistort_map(self):
        u, v = np.meshgrid(np.arange(self.width), np.arange(self.height))
        uv = np.concatenate([u[..., None], v[..., None]], axis=-1)
        return uv[..., 0], uv[..., 1]
