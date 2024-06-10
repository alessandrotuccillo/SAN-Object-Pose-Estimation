import numpy as np


def depth_to_point_cloud(depth, rgb, fx, fy, cx, cy):
    height, width = depth.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    
    # 3D point coordinates
    z = depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    #valid = (z>0)

    # Stack into 3D point cloud
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 4)
    
    return points, colors

def depth_to_point_cloud_augmented(depth, sem_seg, fx, fy, cx, cy):
    height, width = depth.shape
    i, j = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    
    # 3D point coordinates
    z = depth
    x = (i - cx) * z / fx
    y = (j - cy) * z / fy
    label = sem_seg

    # Stack into augmented 3D point cloud
    points = np.stack((x, y, z, sem_seg), axis=-1).reshape(-1, 4)
    
    return points

def augment_point_cloud(point_cloud, sem_seg):
    augmented_point_cloud = np.hstack((point_cloud, sem_seg.reshape(-1, 1)))
    return augmented_point_cloud

def transform_point_cloud(point_cloud, pose):
    # Convert point cloud to homogeneous coordinates
    ones = np.ones((point_cloud.shape[0], 1))
    homogeneous_points = np.hstack((point_cloud, ones))
    
    # Apply transformation
    transformed_points = (pose @ homogeneous_points.T).T
    transformed_points = transformed_points[:, :3]

    # Transformation to world frame
    frame_yaw = 90
    frame_pitch = 180

    # Convert angles from degrees to radians
    yaw_rad = np.deg2rad(frame_yaw)
    pitch_rad = np.deg2rad(frame_pitch)

    # Create rotation matrix using yaw and pitch
    R_yaw = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])
    R_pitch = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])
    camera_rotation = R_pitch @ R_yaw

    world_points = camera_rotation @ transformed_points.T

    transl_int = pose @ np.array([0, 0, 1.2, 1])
    transl_int = transl_int[:3]
    translation = camera_rotation @ transl_int

    world_points_translated = world_points.T - translation

    return world_points_translated

def project_points(points, fx, fy, cx, cy):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    u = (x * fx / z + cx).astype(int)
    v = (y * fy / z + cy).astype(int)
    
    return u, v

def get_point_cloud_extrema(point_cloud, classes):
    class_extremes = {}
    for i in range(len(classes)):
        class_points = point_cloud[point_cloud[:, 3] == i]
        min_coords = class_points[:, :3].min(axis=0)
        max_coords = class_points[:, :3].max(axis=0)
        class_extremes[i] = {
            'min_coords': min_coords,
            'max_coords': max_coords
        }
    return class_extremes

def get_object_position_all(point_cloud, classes):
    class_positions = {}
    for i in range(len(classes)):
        class_points = point_cloud[point_cloud[:, 3] == i]
        mean_coords = class_points[:, :3].mean(axis=0)
        class_positions[i] = mean_coords
    return class_positions

def get_object_position(point_cloud):
    np_points = np.asarray(point_cloud.points)
    mean_coords = np_points[:, :3].mean(axis=0)
    return mean_coords

def get_object_position_from_extrema(point_cloud, classes, class_extremes):
    class_positions = {}
    for i in range(len(classes)):
        class_points = point_cloud[point_cloud[:, 3] == i]
        x_coord = (class_extremes[i]['min_coords'][0] + class_extremes[i]['max_coords'][0]) / 2
        y_coord = (class_extremes[i]['min_coords'][1] + class_extremes[i]['max_coords'][1]) / 2
        z_coord = (class_extremes[i]['min_coords'][2] + class_extremes[i]['max_coords'][2]) / 2
        class_positions[i] = np.array([x_coord, y_coord, z_coord])
    return class_positions


# Function to calculate distances between pairs of points
def calculate_box_edges(corners):
    edges = [
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[1] - corners[2]),
        np.linalg.norm(corners[2] - corners[3]),
        np.linalg.norm(corners[3] - corners[0]),
        np.linalg.norm(corners[4] - corners[5]),
        np.linalg.norm(corners[5] - corners[6]),
        np.linalg.norm(corners[6] - corners[7]),
        np.linalg.norm(corners[7] - corners[4]),
        np.linalg.norm(corners[0] - corners[4]),
        np.linalg.norm(corners[1] - corners[5]),
        np.linalg.norm(corners[2] - corners[6]),
        np.linalg.norm(corners[3] - corners[7]),
    ]
    return edges
'''

def calculate_box_edges(corners):
    edges = [
        np.linalg.norm(corners[0] - corners[1]),
        np.linalg.norm(corners[1] - corners[3]),
        np.linalg.norm(corners[2] - corners[3]),
        np.linalg.norm(corners[0] - corners[2]),
        np.linalg.norm(corners[0] - corners[4]),
        np.linalg.norm(corners[4] - corners[5]),
        np.linalg.norm(corners[5] - corners[7]),
        np.linalg.norm(corners[6] - corners[7]),
        np.linalg.norm(corners[4] - corners[6]),
        np.linalg.norm(corners[1] - corners[5]),
        np.linalg.norm(corners[2] - corners[6]),
        np.linalg.norm(corners[3] - corners[7]),
    ]
    return edges
'''
'''
if __name__ == "__main__":

    # Load the RGB and Depth images
    rgb_image = Image.open('/home/alessandro/SAN/datasets/narrate/0340_57.png')
    depth_image = Image.open('/home/alessandro/SAN/datasets/640x480_dataset/depth/0340_57.png')

    # Convert to numpy arrays
    rgb_image = np.array(rgb_image)
    depth_image = np.array(depth_image)

    # Camera intrinsic parameters
    intrinsic = np.loadtxt("/home/alessandro/SAN/datasets/640x480_dataset/640x480_intrinsic_640x480.txt")
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # Camera pose (rotation and translation matrices)
    # Assuming pose is given as a 4x4 transformation matrix
    camera_pose = np.loadtxt("/home/alessandro/SAN/datasets/640x480_dataset/pose/0340_57.txt")

    point_cloud, colors = depth_to_point_cloud(depth_image, rgb_image, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    o3d.visualization.draw_geometries([pcd])

    transformed_point_cloud = transform_point_cloud(point_cloud, camera_pose)

    
    # Create Open3D Point Cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(transformed_point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    u, v = project_points(transformed_point_cloud, fx, fy, cx, cy)

    # Filter valid points
    valid_indices = (u >= 0) & (u < rgb_image.shape[1]) & (v >= 0) & (v < rgb_image.shape[0])
    u = u[valid_indices]
    v = v[valid_indices]

    # Create a blank image and plot the projected points
    projected_image = np.zeros_like(rgb_image)
    projected_image[v, u] = rgb_image[v, u]

    cv2.imshow('Projected Image', projected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''