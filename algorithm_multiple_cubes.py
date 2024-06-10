from predict import Predictor
import new_algorithm_pointcloud as pc
import cv2
import open3d as o3d
from PIL import Image
import numpy as np

class Object:
    def __init__(self, class_id, name, position, size, rotation, bounding_box, bounding_sphere):
        self.class_id = class_id                # int
        self.name = name                        # str
        self.position = position                # np.array([x, y, z]) - float
        self.size = size                        # float
        self.rotation = rotation                # np.array([rx, ry, rz]) or float ?
        self.bounding_box = bounding_box        # o3d.OrientedBoundingBox
        self.bounding_sphere = bounding_sphere  # o3d.TriangleMesh

def extract_clusters(point_cloud, eps=0.02, min_points=10):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    clusters = []
    for i in range(max_label + 1):
        cluster = point_cloud.select_by_index(np.where(labels == i)[0])
        clusters.append(cluster)
    return clusters, labels

def select_cluster(clusters):
    selected_cluster = None
    max_points = 0

    for cluster in clusters:
        points = np.asarray(cluster.points)
        # Choose largest cluster by number of points
        if len(points) > max_points:
            max_points = len(points)
            selected_cluster = cluster

    return selected_cluster

def run(config_file, model_path, img_path, depth_path, vocab, output_file) -> list[Object]:
    
    vocab = vocab.split(", ")
        
    # Initialize a dictionary to count the occurrences of each object type
    object_counts = {}
    object_items = {}
    
    for item in vocab:
        # Split the object string by '-'
        parts = item.split('_')
        if len(parts) == 2:  # Check if the object string has color and type
            _, obj_type = parts
            # Increment the count for the object type in the dictionary
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
            object_items[obj_type] = object_items.get(obj_type, []) + [item]
        else:
            # Increment the count for the object type in the dictionary
            object_counts[item] = object_counts.get(item, 0) + 1
            object_items[item] = object_items.get(item, []) + [item]

    input_vocab = list(object_counts.keys())
    
    # Segment image
    predictor = Predictor(config_file=config_file, model_path=model_path)
    result = predictor.predict(
        img_path,
        input_vocab,
        augment_vocabulary=False,
        output_file=output_file,
    )

    # Load the RGB and Depth images
    rgb_image = Image.open(img_path)
    depth_image = Image.open(depth_path)

    # Convert to numpy arrays
    rgb_image = np.array(rgb_image)
    depth_image = np.array(depth_image)
    depth_image = depth_image / 10000
    depth_image[depth_image > 2] = 0

    # Camera intrinsic parameters
    fov_x, fov_y = 60, 60
    fx = rgb_image.shape[0] / (2 * np.tan(np.deg2rad(fov_x) / 2))
    fy = rgb_image.shape[1] / (2 * np.tan(np.deg2rad(fov_y) / 2))
    cx, cy = rgb_image.shape[0] / 2, rgb_image.shape[1] / 2

    # Camera view matrix
    view_matrix = (0.7071067690849304, -0.35355344414711, 0.6123724579811096, 0.0, 0.7071068286895752, 0.3535533547401428, -0.6123724579811096, 0.0, 5.960464477539063e-08, 0.866025447845459, 0.5, 0.0, -3.576278473360617e-08, 5.960464477539063e-08, -1.1999999284744263, 1.0)
    view_matrix = np.array(view_matrix).reshape(4, 4)

    # Convert depth image to point cloud
    point_cloud, colors = pc.depth_to_point_cloud(depth_image, rgb_image, fx, fy, cx, cy)
    world_point_cloud = pc.transform_point_cloud(point_cloud, view_matrix)
    augmented_point_cloud = pc.augment_point_cloud(world_point_cloud, result['sem_seg'])
    
    # Visualize the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3] / 255.0)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, coordinate_frame])

    # Visualize the world point cloud
    pcdworld = o3d.geometry.PointCloud()
    pcdworld.points = o3d.utility.Vector3dVector(world_point_cloud)
    pcdworld.colors = o3d.utility.Vector3dVector(colors[:, :3] / 255.0)
    world_coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    
    blue_cube_truepos = [0.1, -0.3, 0.03]
    green_cube_truepos = [-0.1, -0.1, 0.03]
    orange_cube_truepos = [-0.1, -0.1, 0.03]
    red_cube_truepos = [-0.1, -0.3, 0.03]

    blue_cube_true = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
    green_cube_true = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
    orange_cube_true = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
    red_cube_true = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)

    blue_cube_true.translate(blue_cube_truepos)
    green_cube_true.translate(green_cube_truepos)
    orange_cube_true.translate(orange_cube_truepos)
    red_cube_true.translate(red_cube_truepos)

    blue_cube_true.paint_uniform_color([0, 1, 0])
    green_cube_true.paint_uniform_color([0, 1, 0])
    orange_cube_true.paint_uniform_color([0, 1, 0])
    red_cube_true.paint_uniform_color([0, 1, 0])
    
    o3d.visualization.draw_geometries([pcdworld, world_coordinate_frame, blue_cube_true, green_cube_true, orange_cube_true, red_cube_true])
    
    obbs = []
    obb_corners = []
    positions = []
    spheres = []
    bspheres = []
    size = []
    z_rotations = []
    #objects = []
    objects = {}
    positions_fun = []
    spheres_fun = []
    
    for i, item_without_color in enumerate(result['vocabulary']):
        for item in object_items[item_without_color]:

            # Segment the point cloud for each object
            point_cloud_object = world_point_cloud[augmented_point_cloud[:, 3] == i]
            new_color = colors[augmented_point_cloud[:, 3] == i]
            pcdob = o3d.geometry.PointCloud()
            pcdob.points = o3d.utility.Vector3dVector(point_cloud_object)
            pcdob.colors = o3d.utility.Vector3dVector(new_color[:, :3] / 255.0)
        
            # Apply statistical outlier removal
            _, ind = pcdob.remove_statistical_outlier(nb_neighbors=800, std_ratio=0.05) #should make a check on the number of points depending on how many clusters output
            inlier_cloud = pcdob.select_by_index(ind)

            # Extract the clusters and labels
            clusters, labels = extract_clusters(inlier_cloud)

            # Define the color dictionary
            color_to_rgb = {
                "blue": np.array([0, 0, 1]),
                "green": np.array([0, 1, 0]),
                "orange": np.array([1, 0.5, 0]),
                "red": np.array([1, 0, 0]),
                "yellow": np.array([1, 1, 0]),
                "purple": np.array([0.5, 0, 0.5])
            }

            # Function to get the RGB array for a given color name
            def get_rgb_from_color_name(color_name: str):
                return color_to_rgb.get(color_name.lower(), None)

            def get_dominant_color(cluster):
                # Convert colors to numpy array
                colors = np.asarray(cluster.colors)
                # Compute the mean color of the cluster
                mean_color = np.mean(colors, axis=0)
                return mean_color

            unique_labels = np.unique(labels[labels >= 0])
            # Compute the dominant color for each cluster
            dominant_colors = [get_dominant_color(cluster) for cluster in clusters]

            # Find the labels closest to color of item
            color_label = unique_labels[np.argmin([np.linalg.norm(color - get_rgb_from_color_name(item.split('_')[0])) for color in dominant_colors])]

            # Separate the clusters
            color_cluster = inlier_cloud.select_by_index(np.where(labels == color_label)[0])

            # Visualize the result
            #o3d.visualization.draw_geometries([color_cluster])

            # Extract the clusters and select the biggest one
            clusters_2, _ = extract_clusters(color_cluster)
            inlier_cloud = select_cluster(clusters_2)

            # Get oriented bounding box
            obb = inlier_cloud.get_minimal_oriented_bounding_box(True)
            obb.color = (1, 0, 0)  # Red color for the bounding box

            # Get the maximum extent to make the bounding box a cube
            max_extent = max(obb.extent)
            center = obb.center
            rotation_matrix = obb.R

            # Create a cube bounding box
            cube_obb = o3d.geometry.OrientedBoundingBox(center, rotation_matrix, 
                                                        [max_extent, max_extent, max_extent])
            cube_obb.color = (1, 0, 0)  # Red color for the cube bounding box
            obbs.append(cube_obb)

            # Get bounding box corner coordinates
            cube_obb_corner = np.asarray(cube_obb.get_box_points())
            obb_corners.append(cube_obb_corner)

            # Get edges of bounding box
            size.append(max_extent)

            # Get rotation around z-axis (yaw) of bounding box
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            yaw = np.rad2deg(yaw)
            z_rotations.append(yaw)
            
            # Get object position as OBB center
            pos = center
            #print("center:", center)
            positions.append(pos)
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
            sphere.translate(pos)
            sphere.paint_uniform_color([1, 0, 0])  # Red color for spheres
            spheres.append(sphere)

            # Get object position from function (slightly different from OBB center)
            pos_fun = pc.get_object_position(inlier_cloud)
            #print("pos_fun:", pos_fun)
            positions_fun.append(pos_fun)
            sphere_fun = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere_fun.translate(pos)
            sphere_fun.paint_uniform_color([0, 1, 0])  # ? color for spheres
            spheres_fun.append(sphere)

            # Visualize object point cloud with bounding box and centroid
            o3d.visualization.draw_geometries([inlier_cloud, cube_obb, sphere])

            # Get radius of bounding sphere
            distances = np.linalg.norm(np.asarray(inlier_cloud.points) - pos, axis=1)
            radius = np.max(distances)

            # Create bounding sphere
            bsphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            bsphere.translate(pos)
            bsphere.paint_uniform_color([1, 0 , 0])  # Red color for bounding sphere
            bspheres.append(bsphere)

            # Create object
            object = Object(i, item, pos, max_extent, yaw, None, None)
            objects[item] = object

            # Visualize object point cloud with bounding sphere and centroid
            #o3d.visualization.draw_geometries([inlier_cloud, bsphere, sphere])

            # Visualize object point cloud
            #o3d.visualization.draw_geometries([pcdob])

    # Visualize the whole point cloud with bounding box
    #o3d.visualization.draw_geometries([pcd] + obbs)

    # Visualize the whole point cloud with bounding boxes and centroids
    o3d.visualization.draw_geometries([pcdworld, world_coordinate_frame] + spheres + obbs)
    #o3d.visualization.draw_geometries([pcdworld, world_coordinate_frame, blue_cube_true, green_cube_true, orange_cube_true, red_cube_true] + spheres + obbs)

    o3d.io.write_point_cloud("point_cloud_00197.pcd", pcd)

    # Visualize the whole point cloud with bounding spheres and centroids
    #o3d.visualization.draw_geometries([pcd] + spheres + bspheres)

    return objects
    


if __name__ == "__main__":
       
    config_file = "configs/san_clip_vit_res4_coco.yaml"
    model_path = "output/model.pth"
    img_path = "datasets/narrate/colors-4cubes-small.png"
    depth_path = "datasets/narrate/depth-4cubes-small.png"
    vocab = "blue_cube, orange_cube, green_cube, red_cube" #ricorda di risolvere il problema degli spazi vuoti - OK!
    output_file = "output/visualization_cubes5.jpg"

    #posso aggiungere che bounding box deve stare al di sopra del tavolo

    objects = run(config_file, model_path, img_path, depth_path, vocab, output_file)

    print("Blue_cube:", objects["blue_cube"].position, objects["blue_cube"].size, objects["blue_cube"].rotation)
    print("Green_cube:", objects["green_cube"].position, objects["green_cube"].size, objects["green_cube"].rotation)
    print("Orange_cube:", objects["orange_cube"].position, objects["orange_cube"].size, objects["orange_cube"].rotation)
    print("Red_cube:", objects["red_cube"].position, objects["red_cube"].size, objects["red_cube"].rotation)