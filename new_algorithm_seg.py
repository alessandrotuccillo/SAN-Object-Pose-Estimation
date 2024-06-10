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
        self.position = position                # np.array([x, y, z])
        self.size = size                        # np.array([width, height, depth])
        self.rotation = rotation                # np.array([rx, ry, rz]) or float ?
        self.bounding_box = bounding_box        # o3d.OrientedBoundingBox
        self.bounding_sphere = bounding_sphere  # o3d.TriangleMesh

def color_distance(color1, color2):
    return np.linalg.norm(color1 - color2)

def remove_plane(point_cloud, distance_threshold=0.001, ransac_n=3, num_iterations=1000):
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)
    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)
    
    return outlier_cloud

def extract_clusters(point_cloud, eps=0.02, min_points=10):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(point_cloud.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
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
    predictor = Predictor(config_file=config_file, model_path=model_path)
    result = predictor.predict(
        img_path,
        vocab.split(","),
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

    np.savetxt('array.txt', depth_image, delimiter=',')

    # Camera intrinsic parameters
    #fov_x, fov_y = 1.7320507764816284, 1.7320507764816284
    #fov_x = np.rad2deg(2 * np.arctan2(w, 2 * fx))
    fov_x, fov_y = 60, 60
    fx = rgb_image.shape[0] / (2 * np.tan(np.deg2rad(fov_x) / 2))
    fy = rgb_image.shape[1] / (2 * np.tan(np.deg2rad(fov_y) / 2))
    cx, cy = rgb_image.shape[0] / 2, rgb_image.shape[1] / 2

    #fx, fy = 570.3, 570.3
    #cx, cy = 320, 240

    #intrinsic = np.loadtxt("/home/alessandro/SAN/datasets/640x480_dataset/640x480_intrinsic_640x480.txt")
    #fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    #cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    view_matrix = (0.7071067690849304, -0.35355344414711, 0.6123724579811096, 0.0, 0.7071068286895752, 0.3535533547401428, -0.6123724579811096, 0.0, 5.960464477539063e-08, 0.866025447845459, 0.5, 0.0, -3.576278473360617e-08, 5.960464477539063e-08, -1.1999999284744263, 1.0)
    view_matrix = np.array(view_matrix).reshape(4, 4)

    # Convert depth image to point cloud
    point_cloud, colors = pc.depth_to_point_cloud(depth_image, rgb_image, fx, fy, cx, cy)
    #augmented_point_cloud = pc.depth_to_point_cloud_augmented(depth_image, result['sem_seg'], fx, fy, cx, cy)
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
    
    cube_truepos = [0.1, -0.4, 0.03]
    cube_true = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    cube_true.translate(cube_truepos)
    cube_true.paint_uniform_color([1, 0, 0])  # Red color for spheres
    
    o3d.visualization.draw_geometries([pcdworld, world_coordinate_frame, cube_true])
    

    obbs = []
    obb_corners = []
    positions = []
    spheres = []
    bspheres = []
    sizes = []
    z_rotations = []
    objects = []
    positions_fun = []
    spheres_fun = []

    '''
    for i in range(0, len(result['vocabulary'])):
        # Segment the point cloud for each object
        point_cloud_object = point_cloud[augmented_point_cloud[:, 3] == i]
        new_color = colors[augmented_point_cloud[:, 3] == i]
        pcdob = o3d.geometry.PointCloud()
        pcdob.points = o3d.utility.Vector3dVector(point_cloud_object)
        pcdob.colors = o3d.utility.Vector3dVector(new_color[:, :3] / 255.0)
        
        # Apply statistical outlier removal
        cl, ind = pcdob.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.05)
        inlier_cloud = pcdob.select_by_index(ind)

        # Get oriented bounding box
        obb = inlier_cloud.get_minimal_oriented_bounding_box(True)
        #obb = inlier_cloud.get_axis_aligned_bounding_box()
        obb.color = (1, 0, 0)  # Red color for the bounding box
        obbs.append(obb)

        # Get bounding box corner coordinates
        obb_corner = np.asarray(obb.get_box_points())
        obb_corners.append(obb_corner)

        # Get edges of bounding box
        #edge = pc.calculate_box_edges(obb_corner)
        size = obb.extent
        sizes.append(size)

        # Get rotation around z-axis (yaw) of bounding box
        rotation_matrix = obb.R
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        z_rotations.append(yaw)
        
        # Get object position
        pos = pc.get_object_position(inlier_cloud)
        positions.append(pos)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=80000)
        sphere.translate(pos)
        sphere.paint_uniform_color([1, 0, 0])  # Red color for spheres
        spheres.append(sphere)

        # Visualize object point cloud with bounding box and centroid
        o3d.visualization.draw_geometries([inlier_cloud, obb, sphere])

        # Get radius of bounding sphere
        distances = np.linalg.norm(np.asarray(inlier_cloud.points) - pos, axis=1)
        radius = np.max(distances)

        # Create bounding sphere
        bsphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        bsphere.translate(pos)
        bsphere.paint_uniform_color([1, 0 , 0])  # Red color for bounding sphere
        bspheres.append(bsphere)

        # Create object
        object = Object(i, result['vocabulary'][i], pos, size, yaw, obb, bsphere)
        objects.append(object)

        # Visualize object point cloud with bounding sphere and centroid
        #o3d.visualization.draw_geometries([inlier_cloud, bsphere, sphere])

        # Visualize object point cloud
        #o3d.visualization.draw_geometries([pcdob])
    '''

    #for i, item in range(len(result['vocabulary'])), result['vocabulary']:
    for i, item in enumerate(result['vocabulary']):

        # Segment the point cloud for each object
        point_cloud_object = world_point_cloud[augmented_point_cloud[:, 3] == i]
        new_color = colors[augmented_point_cloud[:, 3] == i]
        pcdob = o3d.geometry.PointCloud()
        pcdob.points = o3d.utility.Vector3dVector(point_cloud_object)
        pcdob.colors = o3d.utility.Vector3dVector(new_color[:, :3] / 255.0)
        
        # Apply statistical outlier removal
        cl, ind = pcdob.remove_statistical_outlier(nb_neighbors=1000, std_ratio=0.05)
        inlier_cloud = pcdob.select_by_index(ind)

        # Remove table plane
        #inlier_cloud = remove_plane(inlier_cloud)

        # Usage of clusters
        clusters, labels = extract_clusters(inlier_cloud)
        #inlier_cloud = select_cluster(clusters)

        # You need to choose the correct cluster corresponding to the cube
        #inlier_cloud = clusters[3]  # Example: Select the first cluster

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
        def get_rgb_from_color_name(color_name):
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

        # Assuming blue is predominantly (0, 0, 1) and green is predominantly (0, 1, 0)
        # Find the labels closest to blue and green
        blue_label = unique_labels[np.argmin([np.linalg.norm(color - get_rgb_from_color_name(item.split('-')[0])) for color in dominant_colors])]
        #blue_label = unique_labels[np.argmin([np.linalg.norm(color - np.array([0, 0, 1])) for color in dominant_colors])]
        #green_label = unique_labels[np.argmin([np.linalg.norm(color - np.array([0, 1, 0])) for color in dominant_colors])]
        #orange_label = unique_labels[np.argmin([np.linalg.norm(color - np.array([1, 0.5, 0])) for color in dominant_colors])]


        # Separate the clusters
        blue_cluster = inlier_cloud.select_by_index(np.where(labels == blue_label)[0])
        #green_cluster = inlier_cloud.select_by_index(np.where(labels == green_label)[0])
        #orange_cluster = inlier_cloud.select_by_index(np.where(labels == orange_label)[0])


        # Visualize the result
        o3d.visualization.draw_geometries([blue_cluster])

        clusters_2, _ = extract_clusters(blue_cluster)
        inlier_cloud = select_cluster(clusters_2)



        '''
        # Convert colors to numpy array
        color_array = np.asarray(inlier_cloud.colors)
        
        # Calculate the dominant color (mean color)
        dominant_color = np.mean(color_array, axis=0)

        # Set a threshold for color distance
        color_distance_threshold = 0.1  # This value can be adjusted based on your requirement
        
        # Identify points whose color deviates significantly from the dominant color
        distances = np.apply_along_axis(color_distance, 1, color_array, dominant_color)
        outlier_indices = np.where(distances > color_distance_threshold)[0]

        # Create a mask to remove outliers
        inlier_indices = np.where(distances <= color_distance_threshold)[0]

        # Select inlier points
        inlier_cloud = inlier_cloud.select_by_index(inlier_indices)
        
        # If you want to visualize or further process the inlier_cloud
        o3d.visualization.draw_geometries([inlier_cloud])
        '''

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
        sizes.append([max_extent, max_extent, max_extent])

        # Get rotation around z-axis (yaw) of bounding box
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        z_rotations.append(yaw)
        
        # Get object position as OBB center
        pos = center
        print("center:", center)
        positions.append(pos)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sphere.translate(pos)
        sphere.paint_uniform_color([1, 0, 0])  # Red color for spheres
        spheres.append(sphere)

        # Get object position from function (slightly different from OBB center)
        pos_fun = pc.get_object_position(inlier_cloud)
        print("pos_fun:", pos_fun)
        positions_fun.append(pos_fun)
        sphere_fun = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
        sphere_fun.translate(pos)
        #sphere_fun.paint_uniform_color([0, 1, 0])  # ? color for spheres
        spheres_fun.append(sphere)

        # Visualize object point cloud with bounding box and centroid
        o3d.visualization.draw_geometries([inlier_cloud, cube_obb, sphere, sphere_fun])

        # Calculate distances
        distance_center = np.linalg.norm(np.array(cube_truepos) - np.array(center))
        distance_pos_fun = np.linalg.norm(np.array(cube_truepos) - np.array(pos_fun))

        # Print distances
        print("Distance between cube_truepos and center:", distance_center)
        print("Distance between cube_truepos and pos_fun:", distance_pos_fun)

        # Get radius of bounding sphere
        distances = np.linalg.norm(np.asarray(inlier_cloud.points) - pos, axis=1)
        radius = np.max(distances)

        # Create bounding sphere
        bsphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        bsphere.translate(pos)
        bsphere.paint_uniform_color([1, 0 , 0])  # Red color for bounding sphere
        bspheres.append(bsphere)

        # Create object
        object = Object(i, result['vocabulary'][i], pos, sizes, yaw, obb, bsphere)
        objects.append(object)

        # Visualize object point cloud with bounding sphere and centroid
        #o3d.visualization.draw_geometries([inlier_cloud, bsphere, sphere])

        # Visualize object point cloud
        #o3d.visualization.draw_geometries([pcdob])

    # Visualize the whole point cloud with bounding box
    #o3d.visualization.draw_geometries([pcd] + obbs)

    # Visualize the whole point cloud with bounding boxes and centroids
    o3d.visualization.draw_geometries([pcdworld, world_coordinate_frame, cube_true] + spheres + obbs)
    o3d.io.write_point_cloud("point_cloud_00197.pcd", pcd)

    # Visualize the whole point cloud with bounding spheres and centroids
    #o3d.visualization.draw_geometries([pcd] + spheres + bspheres)

    '''
    # Create planes
    plane_size = 10  # Define the size of the planes

    # XY plane
    xy_plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=plane_size, depth=0.01)
    xy_plane.translate([-plane_size / 2, -plane_size / 2, 0])
    xy_plane.paint_uniform_color([1, 0, 0])  # Red plane

    # XZ plane
    xz_plane = o3d.geometry.TriangleMesh.create_box(width=plane_size, height=0.01, depth=plane_size)
    xz_plane.translate([-plane_size / 2, 0, -plane_size / 2])
    xz_plane.paint_uniform_color([0, 1, 0])  # Green plane

    # YZ plane
    yz_plane = o3d.geometry.TriangleMesh.create_box(width=0.01, height=plane_size, depth=plane_size)
    yz_plane.translate([0, -plane_size / 2, -plane_size / 2])
    yz_plane.paint_uniform_color([0, 0, 1])  # Blue plane

    # Visualize the point cloud and the planes
    o3d.visualization.draw_geometries([pcd, xy_plane, xz_plane, yz_plane])
    '''

    return objects
    


if __name__ == "__main__":
       
    config_file = "configs/san_clip_vit_res4_coco.yaml"
    model_path = "output/model.pth"

    '''
    # Data 1
    img_path = "datasets/narrate/00197-color.png"
    depth_path = "datasets/narrate/00197-depth.png"
    vocab = "cereal-box,can,cup,bowl"
    output_file = "output/visualization_9.jpg"
    '''
    
    
    '''
    # Data 2
    img_path = "datasets/narrate/00203-color.png"
    depth_path = "datasets/narrate/00203-depth.png"
    vocab = "can,cup,white-bowl,green-bowl,hat"
    output_file = "output/visualization_10.jpg"
    '''

    '''
    # Data 3
    img_path = "datasets/narrate/00349-color.png"
    depth_path = "datasets/narrate/00349-depth.png"
    vocab = "hat,cup,white-plate,bowl,can"
    output_file = "output/visualization_11.jpg"
    '''
    
    
    # Data 4
    img_path = "datasets/narrate/colors-3cubes-1.png"
    depth_path = "datasets/narrate/depth-3cubes-1.png"
    vocab = "green-cube"
    output_file = "output/visualization_greencube.jpg"
    

    run(config_file, model_path, img_path, depth_path, vocab, output_file)

    # vedere se con 1 cubo, l'errore è minore scegliendo cluster massimo o con colore simile