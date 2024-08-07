import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
from pypcd4 import PointCloud

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

        planes.append(line)
        planes.append(inlier_cloud)

    return current_pcd, planes

def main(plypath):
    pcd = o3d.io.read_point_cloud(plypath)

    remain_pcd, planes = plane_detection(pcd)
    for plane in planes:
        plane.paint_uniform_color(np.random.rand(3))

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0,0,0])
    # 可视化结果
    o3d.visualization.draw_geometries([ remain_pcd, *planes,mesh_frame])


if __name__ == '__main__':
    pcd_path = 'data/20240803-183210/166/'
    pcd_files = os.listdir(pcd_path)[13:17]
    pcd_files = [os.path.join(pcd_path, file) for file in pcd_files]
    pc_fields = PointCloud.from_path(pcd_files[0]).fields
    pcd_all = [PointCloud.from_path(file).numpy() for file in pcd_files]
    pcd_all = np.vstack(pcd_all)
    print(f'Load and merge all point clouds, shape: {pcd_all.shape}')
    pcd_all = pcd_all[pcd_all[:, 3] > 160][:, :3]
    print(f'After filter by intensity, point clouds shape: {pcd_all.shape}')

    pcd = o3d.t.geometry.PointCloud(
        o3c.Tensor(pcd_all, device=o3c.Device("CUDA:0"))
    ).to_legacy()
    # o3d.visualization.draw_geometries([pcd])

    remain_pcd, planes = plane_detection(pcd)
    print(f'remain_pcd: {remain_pcd}')
    for plane in planes:
        plane.paint_uniform_color(np.random.rand(3))
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # 可视化结果
    o3d.visualization.draw_geometries([*planes, mesh_frame])

    boundary, mask = planes[1].compute_boundary_points(0.5, 60)
    # cloud = cloud.paint_uniform_color([0.6, 0.6, 0.6])
    boundary = boundary.paint_uniform_color([1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries([boundary.to_legacy()])
