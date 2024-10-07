import argparse
import os
import random
import numpy as np
import open3d as o3d
import open3d.core as o3c
from shapely import LineString
from pypcd4 import PointCloud
from apriltag_calibrate.utils.constant import CALIB_BOARD_PARAMS
from skspatial.objects import Plane


def plane_detection(pcd, tolerance=100):
    current_pcd = pcd
    planes = []
    while len(current_pcd.points) > tolerance:
        plane_model, inliers = current_pcd.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=1000)
        if len(inliers) < tolerance:
            break
        inlier_indices = np.asarray(inliers)
        inlier_cloud = current_pcd.select_by_index(inlier_indices)
        current_pcd = current_pcd.select_by_index(inlier_indices, invert=True)

        normal_vector = plane_model[:3]
        point_in_plane = -normal_vector * plane_model[3] / np.linalg.norm(normal_vector) ** 2
        endpoint = point_in_plane + normal_vector * 2

        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([point_in_plane, endpoint])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])

        planes.append((line, inlier_cloud, normal_vector))

    return current_pcd, planes


def proj_pt_to_plane(point, plane_point, plane_normal):
    unit_plane_normal = plane_normal / np.linalg.norm(plane_normal)
    t = np.dot(unit_plane_normal, plane_point - point)
    projection_point = point + t * unit_plane_normal
    return projection_point


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="path to scene root")
    parser.add_argument("--lidar", required=True, help="lidar")
    return parser.parse_args()


def vis_pointcloud(pc, editing=False, show_axis=False):
    if editing is False:
        geometry = []
        if show_axis:
            new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            geometry.append(new_frame)
        if isinstance(pc, np.ndarray):
            pcd = np_to_o3d(pc)
            geometry.append(pcd)
        elif isinstance(pc, o3d.geometry.PointCloud):
            geometry.append(pc)
        o3d.visualization.draw_geometries(geometry)
        return None
    else:
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        if isinstance(pc, np.ndarray):
            pcd = np_to_o3d(pc)
            vis.add_geometry(pcd)
        else:
            vis.add_geometry(pc)

        if show_axis:
            new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis.add_geometry(new_frame)
        vis.run()
        vis.destroy_window()
        print(f"cropped geometry {vis.get_cropped_geometry()}")
        return vis.get_cropped_geometry()


def np_to_o3d(pc):
    assert isinstance(pc, np.ndarray)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    return pcd


def cos_dis(vec_1, vec_2):
    return np.dot(vec_1, vec_2) / (np.linalg.norm(vec_1) * np.linalg.norm(vec_2))


def fit_plane_least_square(points):
    """
    """
    # pcd = np_to_o3d(points)
    # coeffs, inliners = pcd.fit_plane_prerejective()
    # print(f"plane ...coeffs... {coeffs}")
    # normal = coeffs[0:3] / abs(coeffs[3])
    # intercept = coeffs[4]
    # return normal, intercept
    plane = Plane.best_fit(points)
    print(f"plane... {plane}")
    return plane, 0


def compute_axis(plane_1: Plane, plane_2: Plane, plane_3: Plane):
    """

    """
    line_12 = plane_1.intersect_plane(plane_2)
    line_13 = plane_1.intersect_plane(plane_3)
    line_23 = plane_2.intersect_plane(plane_3)

    origin_1 = plane_3.intersect_line(line_12)
    origin_2 = plane_2.intersect_line(line_13)
    origin_3 = plane_1.intersect_line(line_23)

    print(f'origin_1 {origin_1}, origin_2 {origin_2}, origin_3 {origin_3}')
    print(f"line_12 {line_12}, line_13 {line_13}, line_23 {line_23}")
    norm_3 = np.array([line_12.direction[0], line_12.direction[1], line_12.direction[2]])
    norm_2 = np.array([line_13.direction[0], line_13.direction[1], line_13.direction[2]])
    norm_1 = np.array([line_23.direction[0], line_23.direction[1], line_23.direction[2]])

    print(f"1 X 2  {cos_dis(norm_1, norm_2)}")
    print(f"1 X 3 {cos_dis(norm_1, norm_3)}")
    print(f"2 X 3 {cos_dis(norm_2, norm_3)}")


def align_tr(tr, board_points: np.ndarray):
    """

    """

    points = (tr[:3, :3] @ board_points[:, :3].T + tr[:3, 3:]).T

    new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    new_board_pcd = np_to_o3d(points)
    o3d.visualization.draw_geometries([new_board_pcd, new_frame])

    min_x = np.min(points[:, 0])
    min_y = np.min(points[:, 1])
    min_z = np.min(points[:, 2])
    print(f"..{min_x, min_y, min_z}")
    tr_xyz = np.eye(4)
    thresh = -0.5
    if min_x < thresh and min_y < thresh and min_z > thresh:
        tr_xyz = np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    elif min_x < thresh and min_y > thresh and min_z < thresh:
        tr_xyz = np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    elif min_x > thresh and min_y < thresh and min_z < thresh:
        tr_xyz = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    elif min_x < thresh and min_y < thresh and min_z < thresh:
        tr_xyz = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    elif min_x < thresh and min_y > thresh and min_z > thresh:
        tr_xyz = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    elif min_x > thresh and min_y < thresh and min_z > thresh:
        tr_xyz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    elif min_x > thresh and min_y > thresh and min_z < thresh:
        tr_xyz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    print(f"tr_xyz...  {tr_xyz}")
    final_tr = tr_xyz @ tr
    new_box_points = (final_tr[:3, :3] @ board_points.T + final_tr[:3, 3:]).T
    new_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    new_board_pcd = np_to_o3d(new_box_points)
    o3d.visualization.draw_geometries([new_board_pcd, new_frame])
    return final_tr


def manual_check_axis(points: np.ndarray):
    """

    """
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    tr = np.eye(4)
    pcd_1 = np_to_o3d(points)
    vis.add_geometry(pcd_1)
    # x, y, z, red, green, blue
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(mesh)

    def compute_rotate_pointcloud():
        nonlocal tr
        pc = (tr[:3, :3] @ points[:, :3].T + tr[:3, 3:]).T
        pcd = np_to_o3d(pc)
        return pcd
    def show_pointcloud():
        vis.clear_geometries()
        pcd = compute_rotate_pointcloud()
        vis.add_geometry(pcd)
        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(mesh)
        vis.update_renderer()
        vis.poll_events()

    def swap_yz_callback(vis):
        nonlocal tr
        tr_yz = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        tr = tr_yz @ tr
        show_pointcloud()
        return True

    def swap_xz_callback(vis):
        nonlocal tr
        tr_xz = np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        tr = tr_xz @ tr
        show_pointcloud()
        return True

    def swap_xy_callback(vis):
        nonlocal tr
        tr_xy = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        tr = tr_xy @ tr
        show_pointcloud()
        return True

    def clear_tr_callback(vis):
        nonlocal tr
        tr = np.eye(4, dtype=np.float32)
        show_pointcloud()
        return True

    vis.register_key_callback(ord('X'), swap_yz_callback)
    vis.register_key_callback(ord('Y'), swap_xz_callback)
    vis.register_key_callback(ord('Z'), swap_xy_callback)
    vis.register_key_callback(ord('R'), clear_tr_callback)

    vis.run()
    return tr


def main(args):
    """

    """
    # pcd_path = '/home/anchen/solutions/data/p1/20240928-2139_dump'
    pcd_path = os.path.join(args.root, args.lidar)
    pcd_files = os.listdir(pcd_path)  # [13:17]
    pcd_files = [os.path.join(pcd_path, file) for file in pcd_files]
    pcd_all = [PointCloud.from_path(file).numpy() for file in pcd_files]
    pcd_all = np.vstack(pcd_all)
    print(f'Load and merge all point clouds, shape: {pcd_all.shape}')
    vis_pointcloud(pcd_all)
    pcd_filtered = pcd_all[pcd_all[:, 3] > 140][:, :3]
    print(f'After filter by intensity, point clouds shape: {pcd_all.shape}')
    pcd_filtered_o3d = np_to_o3d(pcd_filtered)
    picked = vis_pointcloud(pcd_filtered_o3d, editing=True)
    print(f"picked... {picked}")

    print(f"get center ... {picked.get_center()}")
    box_center = np.array(picked.get_center())
    dist = np.linalg.norm(pcd_all[:, :3] - box_center, axis=1)
    mask = dist < 1.0
    box_points = pcd_all[mask]
    board_points = vis_pointcloud(box_points, editing=True)

    obb = board_points.get_minimal_oriented_bounding_box()
    obb.color = (0, 1, 0)
    outlier_pcd, planes = plane_detection(board_points)
    print(f"obb... {obb.get_box_points()}")
    print(f"center... {obb.center}")
    print(f"R... {obb.R}")
    print(f"extent ... {obb.extent}")
    # planes = sorted(planes, key=lambda d: d[0].size(), reverse=True)
    print(f'plane outlier_pcd: {outlier_pcd}')
    axis_1 = obb.R[:, 0]
    axis_2 = obb.R[:, 1]
    axis_3 = obb.R[:, 2]

    plane_1_points = []
    plane_2_points = []
    plane_3_points = []
    for plane in planes:
        norm_vec = plane[2]
        if np.fabs(cos_dis(norm_vec, axis_1)) > 0.9:
            plane_1_points.extend(plane[1].points)
        elif np.fabs(cos_dis(norm_vec, axis_2)) > 0.9:
            plane_2_points.extend(plane[1].points)
        elif np.fabs(cos_dis(norm_vec, axis_3)) > 0.9:
            plane_3_points.extend(plane[1].points)

    # norm fitting.
    plane_1, intercept_1 = fit_plane_least_square(np.asarray(plane_1_points))
    plane_2, intercept_2 = fit_plane_least_square(np.asarray(plane_2_points))
    plane_3, intercept_3 = fit_plane_least_square(np.asarray(plane_3_points))
    line_12 = plane_1.intersect_plane(plane_2)
    origin = plane_3.intersect_line(line_12)
    print(f"origin... {origin}, {line_12}")
    rotation = np.eye(3)
    rotation[:, 0] = plane_1.normal
    rotation[:, 1] = plane_2.normal
    rotation[:, 2] = plane_3.normal
    print(f"rotation... {rotation}")
    u1, s1, v1 = np.linalg.svd(rotation)
    rotation_2 = u1 @ v1
    print(f"rotation_2 ... {rotation_2}")
    if np.linalg.det(rotation_2) < 0:
        u2, s2, v2 = np.linalg.svd(rotation_2)
        print(f"s2... {s2}")
        s2[-1] *= -1
        rotation_2 = u2 @ np.array([[s2[0], 0, 0], [0, s2[1], 0], [0, 0, s2[2]]]) @ v2
        print(f"rectified rotation_2 {rotation_2}")

    tr = np.eye(4, dtype=np.float32)
    tr[:3, :3] = rotation_2.T
    tr[:3, 3] = -rotation_2.T @ np.array([origin[0], origin[1], origin[2]])
    print(f"Tr... {tr}")

    final_tr = align_tr(tr, box_points[:, :3])
    print(f"final_tr ... {final_tr}")
    # based on intensity find the black zone.

    intensity = box_points[:, 3]
    blackhole_mask = intensity < 100
    blackhole = box_points[blackhole_mask]
    # blackhole on xy-plane.
    blackhole_board = (final_tr[:3, :3] @ blackhole[:, :3].T + final_tr[:3, 3:]).T
    # blackhole_pcd = np_to_o3d(blackhole_board)
    extra_rotation = manual_check_axis(blackhole_board)
    # yz_plane = vis_pointcloud(blackhole_pcd, editing=True, show_axis=False)
    # assert isinstance(yz_plane, o3d.geometry.PointCloud)
    # yz_plane_points = np.array(yz_plane.points)
    # print(f"max x {np.max(yz_plane_points[:, 0])}")
    # print(f"max y {np.max(yz_plane_points[:, 1])}")
    # print(f"max z {np.max(yz_plane_points[:, 2])}")
    print(f"extra_rotation {extra_rotation}")
    print(f"final ... {extra_rotation @ final_tr}")


if __name__ == '__main__':
    main(parse_args())
