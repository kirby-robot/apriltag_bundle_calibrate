import argparse
import os
from pypcd4 import PointCloud
import numpy as np
import open3d as o3d
import copy


def load_all_lidar(pcd_folder):
    pcd_files = os.listdir(pcd_folder)  # [13:17]
    pcd_files = [os.path.join(pcd_folder, file) for file in pcd_files]
    if len(pcd_files) > 20:
        pcd_files = pcd_files[:20]
    pcd_all = [PointCloud.from_path(file).numpy() for file in pcd_files]
    pcd_all = np.vstack(pcd_all)
    return pcd_all


LIDAR_154 = "lidar_192_168_1_154"
LIDAR_166 = "lidar_192_168_1_166"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    return parser.parse_args()


def np_to_o3d(pc):
    assert isinstance(pc, np.ndarray)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    return pcd


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def main(args):
    cloud_154 = load_all_lidar(os.path.join(args.root, LIDAR_154))
    cloud_166 = load_all_lidar(os.path.join(args.root, LIDAR_166))

    # tr_154_2_166 = np.array([[-0.36341149, 0.69734635, -0.61777032, 7.12036024],
    #                          [0.88328148, 0.04706467, -0.46647477, 2.59959878],
    #                          [0.29621933, 0.71518736, 0.63305697, -6.4482031],
    #                          [0., 0., 0., 1.]])
    rotate = np.array([[1, 0, 0, -0.8],
                       [0, -1, 0, 9.5],
                       [0, 0, -1, 7.0],
                       [0, 0, 0, 1]])
    # init_guess = rotate @ tr_154_2_166

    # init_guess = tr_154_2_166
    init_guess = rotate

    pcd_154 = np_to_o3d(cloud_154)
    pcd_166 = np_to_o3d(cloud_166)

    pcd_154.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_166.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    draw_registration_result(pcd_154, pcd_166, init_guess)
    threshold = 0.1
    evaluation = o3d.pipelines.registration.evaluate_registration(pcd_154, pcd_166, threshold, init_guess)
    print(f"evaluation... {evaluation}")
    icp_p2plane = o3d.pipelines.registration.registration_icp(pcd_154, pcd_166, threshold, init_guess,
                                                              o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(1000))
    print(f"icp_p2plane.. {icp_p2plane}")

    print(f"transformation: {icp_p2plane.transformation}")
    draw_registration_result(pcd_154, pcd_166, icp_p2plane.transformation)
    pcd_154.transform(icp_p2plane.transformation)
    o3d.io.write_point_cloud("154.pcd", pcd_154)
    o3d.io.write_point_cloud("166.pcd", pcd_166)
    pcd_all = pcd_166 + pcd_154
    o3d.io.write_point_cloud("icp_154_166.pcd", pcd_all)


if __name__ == '__main__':
    main(parse_args())
