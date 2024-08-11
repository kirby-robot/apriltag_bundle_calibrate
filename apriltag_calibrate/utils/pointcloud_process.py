import os
import random
import numpy as np
import open3d as o3d
import open3d.core as o3c
from shapely import LineString
from pypcd4 import PointCloud
from utils.constant import CALIB_BOARD_PARAMS

def plane_detection(pcd, tolerance = 50):
    current_pcd = pcd
    planes = []
    while len(current_pcd.points) > tolerance:
        plane_model, inliers = current_pcd.segment_plane(distance_threshold=0.1, ransac_n=5, num_iterations=1000)
        if len(inliers) < tolerance:
            break
        inlier_indices = np.asarray(inliers)
        inlier_cloud = current_pcd.select_by_index(inlier_indices)
        current_pcd = current_pcd.select_by_index(inlier_indices, invert=True)

        normal_vector = plane_model[:3]
        point_in_plane = -normal_vector * plane_model[3] / np.linalg.norm(normal_vector)**2
        endpoint = point_in_plane + normal_vector * 2

        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([point_in_plane, endpoint])
        line.lines = o3d.utility.Vector2iVector([[0, 1]])

        planes.append((line, inlier_cloud, normal_vector))

    return current_pcd, planes

class Line:
    """
    Implementation for 3D Line RANSAC.

    This object finds the equation of a line in 3D space using RANSAC method.
    This method uses 2 points from 3D space and computes a line. The selected candidate will be the line with more inliers inside the radius theshold.

    ![3D line](https://raw.githubusercontent.com/leomariga/pyRANSAC-3D/master/doc/line.gif "3D line")

    ---
    """

    def __init__(self):
        self.inliers = []
        self.A = []
        self.B = []

    def fit(self, pts, thresh=0.2, maxIteration=1000):
        """
        Find the best equation for the 3D line. The line in a 3d enviroment is defined as y = Ax+B, but A and B are vectors intead of scalars.

        :param pts: 3D point cloud as a `np.array (N,3)`.
        :param thresh: Threshold distance from the line which is considered inlier.
        :param maxIteration: Number of maximum iteration which RANSAC will loop over.
        :returns:
        - `A`: 3D slope of the line (angle) `np.array (1, 3)`
        - `B`: Axis interception as `np.array (1, 3)`
        - `inliers`: Inlier's index from the original point cloud. `np.array (1, M)`
        ---
        """
        n_points = pts.shape[0]
        best_inliers = []

        for _ in range(maxIteration):
            # Samples 2 random points
            id_samples = random.sample(range(0, n_points), 2)
            pt_samples = pts[id_samples]

            # The line defined by two points is defined as P2 - P1
            vecA = pt_samples[1, :] - pt_samples[0, :]
            vecA_norm = vecA / np.linalg.norm(vecA)

            # Distance from a point to a line
            pt_id_inliers = []  # list of inliers ids
            vecC_stakado = np.stack([vecA_norm] * n_points, 0)
            dist_pt = np.cross(vecC_stakado, (pt_samples[0, :] - pts))
            dist_pt = np.linalg.norm(dist_pt, axis=1)

            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]

            if len(pt_id_inliers) > len(best_inliers):
                best_inliers = pt_id_inliers
                self.inliers = best_inliers
                self.A = vecA_norm
                self.B = pt_samples[0, :]

        return self.A, self.B, self.inliers

def line_fit(pts: np.array):
    center = np.mean(pts, axis=0)
    shift_pts = pts - center
    _, _, vv = np.linalg.svd(shift_pts)
    unit_dir = vv[0] / np.linalg.norm(vv[0])
    proj_length = shift_pts.dot(unit_dir)

    # delete points with large gap
    sorted_length = np.sort(proj_length)
    delta_length = sorted_length - [sorted_length[0], *sorted_length[:-1]]
    max_delta_idx = np.argmax(delta_length)
    if delta_length[max_delta_idx] > 0.15:
        if max_delta_idx < len(delta_length) / 10:
            proj_length = sorted_length[max_delta_idx:]
        elif max_delta_idx > len(delta_length) * 9 / 10:
            proj_length = sorted_length[:max_delta_idx - 1]

    start_pt = proj_length.min() * unit_dir + center
    end_pt = proj_length.max() * unit_dir + center
    return (start_pt, end_pt, proj_length.max() - proj_length.min(), unit_dir)

def line_detection(pts: np.array, thresh: float=0.05, maxIteration: int=1000):
    line_detector = Line()
    lines = list()
    num_pts = pts.shape[0]
    while True:
        dir, point, inliers = line_detector.fit(pts, thresh, maxIteration)
        if len(inliers) < 0.1 * num_pts:
            break
        lines.append(line_fit(pts.take(inliers, axis=0)))
        pts = np.delete(pts, inliers, axis=0)
        if len(pts) < 10:
            break

    return lines

def proj_pt_to_plane(point, plane_point, plane_normal):
    unit_plane_normal = plane_normal / np.linalg.norm(plane_normal)
    t = np.dot(unit_plane_normal, plane_point - point)
    projection_point = point + t * unit_plane_normal
    return projection_point

if __name__ == '__main__':
    pcd_path = 'data/20240803-183320/166/'
    pcd_files = os.listdir(pcd_path)    #[13:17]
    pcd_files = [os.path.join(pcd_path, file) for file in pcd_files]
    pc_fields = PointCloud.from_path(pcd_files[0]).fields
    pcd_all = [PointCloud.from_path(file).numpy() for file in pcd_files]
    pcd_all = np.vstack(pcd_all)
    print(f'Load and merge all point clouds, shape: {pcd_all.shape}')
    pcd_all = pcd_all[pcd_all[:, 3] > 150][:, :3]
    print(f'After filter by intensity, point clouds shape: {pcd_all.shape}')

    pcd = o3d.t.geometry.PointCloud(
        o3c.Tensor(pcd_all, device=o3c.Device("CUDA:0"))
    ).to_legacy()
    # o3d.visualization.draw_geometries([pcd])

    outlier_pcd, planes = plane_detection(pcd)
    print(f'plane outlier_pcd: {outlier_pcd}')
    # for plane in planes:
    #     plane.paint_uniform_color(np.random.rand(3))
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # o3d.visualization.draw_geometries([*planes, mesh_frame])

    calib_board_pc = o3d.t.geometry.PointCloud.from_legacy(planes[0][1])
    calib_board_pc.estimate_normals(radius=0.05, max_nn=20)
    cl, ind = calib_board_pc.remove_statistical_outliers(nb_neighbors=20, std_ratio=0.4)
    calib_board_pc.point['positions'] = cl.point['positions']
    calib_board_pc.estimate_normals(radius=0.1, max_nn=20)
    print(f'calib_board_pc: {calib_board_pc}')

    # extract boundary, boundary line and corners
    boundary, mask = calib_board_pc.compute_boundary_points(0.5, 30)
    cloud = calib_board_pc.paint_uniform_color([0.6, 0.6, 0.6])
    boundary = boundary.paint_uniform_color([0.0, 0.0, 1.0])
    print(f'boundary: {boundary}')

    boundary_pts = boundary.point['positions'].numpy()
    boundary_lines = line_detection(boundary_pts, thresh=0.02)
    assert len(boundary_lines) >= 4, f'detected boundary lines should be more than or equal to 4'
    for line in boundary_lines:
        print(line)

    line_idx = 0
    line_points = []
    line_connect = []
    for line in boundary_lines:
        line_points.append(line[0])
        line_points.append(line[1])
        line_connect.append([2 * line_idx, 2 * line_idx + 1])
        line_idx += 1

    boundary_lines = sorted(boundary_lines, key=lambda x: x[2], reverse=True)
    boundary_linestrings = []
    calib_plane_norm = planes[0][2]
    calib_plane_center = np.asarray(planes[0][1].points).mean(axis=0)
    for line in boundary_lines:
        # project start and end point of boundary line to calib board plane
        start_pt = proj_pt_to_plane(line[0], calib_plane_center, calib_plane_norm)
        end_pt = proj_pt_to_plane(line[1], calib_plane_center, calib_plane_norm)
        unit_dir = end_pt - start_pt
        unit_dir = unit_dir / np.linalg.norm(unit_dir)
        boundary_linestrings.append(LineString([(start_pt - 1.0 * unit_dir).tolist(), (end_pt + 1.0 * unit_dir).tolist()]))

        line_points.append(start_pt)
        line_points.append(end_pt)
        line_connect.append([2 * line_idx, 2 * line_idx + 1])
        line_idx += 1
    corner1 = np.asarray(boundary_linestrings[0].intersection(boundary_linestrings[1]).coords).squeeze(0)
    corner2 = np.asarray(boundary_linestrings[2].intersection(boundary_linestrings[3]).coords).squeeze(0)
    print(f'corner1: {corner1}, corner2: {corner2}')

    line_points.extend([corner1, corner2])
    line_connect.append([len(line_points) - 2, len(line_points) - 1])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector(line_connect)

    # use boundary and corner to generate init transform
    init_trans = corner2
    init_rotation = np.eye(3, dtype=np.float32)
    if np.linalg.norm(corner2 - boundary_lines[3][0]) < np.linalg.norm(corner2 - boundary_lines[3][1]):
        init_rotation[:3, 0] = -boundary_lines[3][3]
        init_trans += boundary_lines[3][3] * CALIB_BOARD_PARAMS['board_size'][0]
    else:
        init_rotation[:3, 0] = boundary_lines[3][3]
        init_trans -= boundary_lines[3][3] * CALIB_BOARD_PARAMS['board_size'][0]
    if np.linalg.norm(corner2 - boundary_lines[2][0]) < np.linalg.norm(corner2 - boundary_lines[2][1]):
        init_rotation[:3, 2] = -boundary_lines[2][3]
        init_trans += boundary_lines[2][3] * CALIB_BOARD_PARAMS['board_size'][1]
    else:
        init_rotation[:3, 2] = boundary_lines[2][3]
        init_trans -= boundary_lines[2][3] * CALIB_BOARD_PARAMS['board_size'][1]
    init_rotation[:3, 1] = np.cross(init_rotation[:3, 2], init_rotation[:3, 0])
    init_transform = np.eye(4)
    init_transform[:3, :3] = init_rotation
    init_transform[:3, 3] = init_trans

    # generate calib board edge point cloud and do icp to refine transform
    calib_board_pc = list()
    calib_board_pc.extend([[1.0, 0.0, z] for z in np.arange(0, 1.0, 0.01)])
    calib_board_pc.extend([[x, 0.0, 1.0] for x in np.arange(0, 1.0, 0.01)])
    calib_board_pc.extend([[CALIB_BOARD_PARAMS['board_size'][0], 0.0, z] for z in np.arange(0, CALIB_BOARD_PARAMS['board_size'][1], 0.01)])
    calib_board_pc.extend([[x, 0.0, CALIB_BOARD_PARAMS['board_size'][1]] for x in np.arange(0, CALIB_BOARD_PARAMS['board_size'][0], 0.01)])
    calib_board_pc = o3d.t.geometry.PointCloud.from_legacy(o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(calib_board_pc),
    ))
    register_res = o3d.t.pipelines.registration.icp(
        calib_board_pc,
        boundary,
        0.1,
        o3c.Tensor(init_transform),
        criteria=o3d.t.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=0.001,
            relative_rmse=0.1,
            max_iteration=20
        )
    )
    print(f'init transform: {init_transform}\nrefine transform: {register_res.transformation}')

    calib_board_pc = calib_board_pc.transform(register_res.transformation)
    calib_board_pc = calib_board_pc.paint_uniform_color([0.0, 1.0, 0.0])

    # visualize final calib board coordinate
    points = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0]],
            dtype=np.float32
        )
    lines = np.array([
            [0, 1],
            [0, 2],
            [0, 3]],
            dtype=np.int64
    )
    colors = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]],
            dtype=np.float32
    )
    calib_board_coord = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    calib_board_coord.colors = o3d.utility.Vector3dVector(colors)
    calib_board_coord.transform(register_res.transformation.numpy())

    # visualize all elements
    o3d.visualization.draw_geometries([line_set, cloud.to_legacy(), boundary.to_legacy(), calib_board_pc.to_legacy(), calib_board_coord])

