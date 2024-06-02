from predict import Predictor
import new_algorithm_pointcloud as pc
import cv2
import open3d as o3d
from PIL import Image
import numpy as np


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--config-file", type=str, required=True, help="path to config file"
    )
    parser.add_argument(
        "--model-path", type=str, required=True, help="path to model file"
    )
    parser.add_argument(
        "--img-path", type=str, required=True, help="path to image file."
    )
    parser.add_argument("--aug-vocab", action="store_true", help="augment vocabulary.")
    parser.add_argument(
        "--vocab",
        type=str,
        default="",
        help="list of category name. seperated with ,.",
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="path to output file."
    )
    args = parser.parse_args()
    #le chiamate a img_path, vocab, output file devono essere cambiate
    #perché non saranno date da linea di comando ma internamente al codice
    predictor = Predictor(config_file=args.config_file, model_path=args.model_path)
    result = predictor.predict(
        args.img_path,
        args.vocab.split(","),
        args.aug_vocab,
        output_file=args.output_file,
    ) #gives sem_seg and vocabulary as output
    #print("result: ", result)

    # python new_algorithm.py --config-file configs/san_clip_vit_res4_coco.yaml --model-path output/model.pth --img-path datasets/narrate/0372_27.png --aug-vocab "COCO-all" --vocab vase,banana,orange  --output-file output/visualization_4.jpg

    # Load the RGB and Depth images
    rgb_image = Image.open('/home/alessandro/SAN/datasets/narrate/00197-color.png')
    depth_image = Image.open('/home/alessandro/SAN/datasets/narrate/00197-depth.png')

    # Convert to numpy arrays
    rgb_image = np.array(rgb_image)
    depth_image = np.array(depth_image)

    # Camera intrinsic parameters
    intrinsic = np.loadtxt("/home/alessandro/SAN/datasets/640x480_dataset/640x480_intrinsic_640x480.txt")
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    fx, fy = 570.3, 570.3
    cx, cy = 320, 240

    # Camera pose (rotation and translation matrices)
    # Assuming pose is given as a 4x4 transformation matrix
    camera_pose = np.loadtxt("/home/alessandro/SAN/datasets/640x480_dataset/pose/0372_27.txt")

    point_cloud, colors = pc.depth_to_point_cloud(depth_image, rgb_image, fx, fy, cx, cy)
    augmented_point_cloud = pc.depth_to_point_cloud_augmented(depth_image, result['sem_seg'], fx, fy, cx, cy)
    #print("augmented_point_cloud: ", augmented_point_cloud)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    o3d.visualization.draw_geometries([pcd])

    obbs = []
    obb_corners = []
    positions = []
    spheres = []
    bspheres = []
    edges = []
    for i in range(0, len(result['vocabulary'])):
        # Segment the point cloud for each object
        point_cloud_object = point_cloud[augmented_point_cloud[:, 3] == i]
        new_color = colors[augmented_point_cloud[:, 3] == i]
        pcdob = o3d.geometry.PointCloud()
        pcdob.points = o3d.utility.Vector3dVector(point_cloud_object)
        pcdob.colors = o3d.utility.Vector3dVector(new_color / 255.0)
        
        # Apply statistical outlier removal
        cl, ind = pcdob.remove_statistical_outlier(nb_neighbors=100, std_ratio=0.05)
        inlier_cloud = pcdob.select_by_index(ind)

        # Get oriented bounding box
        obb = inlier_cloud.get_oriented_bounding_box()
        obb.color = (1, 0, 0)  # Red color for the bounding box
        obbs.append(obb)

        # Get bounding box corner coordinates
        obb_corner = np.asarray(obb.get_box_points())
        obb_corners.append(obb_corner)

        # Get edges of bounding box
        edge = pc.calculate_box_edges(obb_corner)
        edges.append(edge)
        
        # Get object position
        pos = pc.get_object_position(inlier_cloud)
        positions.append(pos)
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=50)
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

        # Visualize object point cloud with bounding sphere and centroid
        #o3d.visualization.draw_geometries([inlier_cloud, bsphere, sphere])

        # Visualize object point cloud
        #o3d.visualization.draw_geometries([pcdob])

    # Visualize the whole point cloud with bounding box
    #o3d.visualization.draw_geometries([pcd] + obbs)

    # Visualize the whole point cloud with bounding boxes and centroids
    o3d.visualization.draw_geometries([pcd] + spheres + obbs)

    # Visualize the whole point cloud with bounding spheres and centroids
    #o3d.visualization.draw_geometries([pcd] + spheres + bspheres)


    '''
    # Create planes
    plane_size = 1000  # Define the size of the planes

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

    # Transform the point cloud
    transformed_point_cloud = pc.transform_point_cloud(point_cloud, camera_pose)
    '''

    '''
    point_cloud_extrema = pc.get_point_cloud_extrema(augmented_point_cloud, result['vocabulary'])
    class_positions = pc.get_object_position_from_extrema(augmented_point_cloud, result['vocabulary'], point_cloud_extrema)
    print("class_positions: ", class_positions)
    
    spheres = []
    for class_id, position in class_positions.items():
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=50)
        sphere.translate(position)
        sphere.paint_uniform_color([1, 0, 0])  # Red color for spheres
        spheres.append(sphere)

    # Visualize the point cloud, planes, and class position spheres
    o3d.visualization.draw_geometries([pcd] + spheres)

    # Function to compute the 8 corners of the bounding box
    def compute_bounding_box_corners(point_cloud_extrema, classes):
        corners = {}
        for i in range(len(classes)):
            min_x = point_cloud_extrema[i]['min_coords'][0]
            max_x = point_cloud_extrema[i]['max_coords'][0]
            min_y = point_cloud_extrema[i]['min_coords'][1]
            max_y = point_cloud_extrema[i]['max_coords'][1]
            min_z = point_cloud_extrema[i]['min_coords'][2]
            max_z = point_cloud_extrema[i]['max_coords'][2]
            corners[i] = np.array([
                [min_x, min_y, min_z], #1
                [min_x, min_y, max_z], #2
                [min_x, max_y, min_z], #3
                [min_x, max_y, max_z], #4
                [max_x, min_y, min_z], #5
                [max_x, min_y, max_z], #6
                [max_x, max_y, min_z], #7
                [max_x, max_y, max_z] #8
            ])
        return corners

    # Compute the corners
    corners = compute_bounding_box_corners(point_cloud_extrema, result['vocabulary'])

    line_sets = []
    for class_id, corner in corners.items():
        print("class_id", class_id)
        print("corner", corner)
        # Create lines for the bounding box edges
        lines = np.array([
            [0, 2], [2, 6], [6, 4], [4, 0],  # Bottom face
            [1, 3], [3, 7], [7, 5], [5, 1],  # Top face
            [0, 1], [2, 3], [4, 5], [6, 7]   # Side edges
        ])

        # Define colors for the lines (optional)
        colors = [[1, 0, 0] for _ in range(len(lines))]  # Red color for the bounding box lines

        # Create the line set for the bounding box
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(corners[class_id])
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        line_sets.append(line_set)

    # Visualize the point cloud and bounding box
    o3d.visualization.draw_geometries([pcd] + line_sets)
    '''