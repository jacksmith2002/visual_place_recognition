import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import habitat_sim
import cv2
import requests
from io import BytesIO
from torchvision import transforms
import math

# from habitat_sim.utils.data import ImageExtractor, PoseExtractor
import magnum as mn
from sklearn.cluster import KMeans


def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)

    # For viewing the extractor output


def display_extractor_sample(sample):
    img = sample["rgba"]

    arr = [img]
    titles = ["rgba"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()


def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    # sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    color_sensor_spec.hfov = settings["sensor_hfov"]
    sensor_specs.append(color_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.1)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def navigateAndSee(action=""):
    if action in action_names:
        observations = sim.step(action)
        print("action: ", action)
        if display:
            display_sample(observations["color_sensor"])


def print_scene_recur(scene, limit_output=10):
    print(
        f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
    )
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    count = 0
    for level in scene.levels:
        print(
            f"Level id:{level.id}, center:{level.aabb.center},"
            f" dims:{level.aabb.sizes}"
        )
        for region in level.regions:
            print(
                f"Region id:{region.id}, category:{region.category.name()},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            for obj in region.objects:
                print(
                    f"Object id:{obj.id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                count += 1
                if count >= limit_output:
                    return


# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown


# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=10, alpha=0.8)
    plt.show(block=False)


import numpy as np
import matplotlib.pyplot as plt


def create_map_image(topdown_map, key_points=None):
    """
    Creates an image of the topdown map with optional keypoints overlaid.

    Args:
    - topdown_map (np.array): The map to visualize.
    - key_points (list of tuples): Optional points to plot on the map.

    Returns:
    - img_array (np.array): An array representing the RGB image of the map.
    """
    # Create a figure with an Axes to plot on
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")  # Hide axes

    # Display the map
    ax.imshow(topdown_map, aspect="auto")

    # If keypoints are provided, plot them
    if key_points is not None:
        x, y = zip(*key_points)  # Unpack points into x and y coordinates
        ax.scatter(x, y, c="red", s=100, alpha=0.8)  # Plot keypoints

    # Save the current figure to an image array
    fig.canvas.draw()  # Draw the figure to capture the image data
    img_array = np.array(fig.canvas.renderer.buffer_rgba())

    # Close the figure to free memory
    plt.close(fig)

    # Convert RGBA to RGB
    img_array = img_array[..., :3]

    return img_array


def create_video_from_image_arrays(image_arrays, output_path, fps=60):
    """
    Creates a video from a list of numpy arrays representing images.

    Args:
    - image_arrays (list): List of numpy arrays of images.
    - output_path (str): Path where the output video will be saved.
    - fps (int): Frames per second of the output video.
    """
    # Assume all images are the same size and take the size from the first image
    height, width, layers = image_arrays[0].shape
    frame_size = (width, height)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    # Loop through the numpy array images
    for img in image_arrays:
        # Ensure the image is in the correct color format
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Write the frame to the video
        out.write(img_bgr)

    # Release everything when job is finished
    out.release()
    print("Video creation complete.")


def convert_topdown_to_3d(pathfinder, topdown_points, meters_per_pixel, agent_height):
    points_3d = []
    bounds = pathfinder.get_bounds()
    for td_point in topdown_points:
        # convert topdown x,y to 3D x,z
        x = td_point[0] * meters_per_pixel + bounds[0][0]
        z = td_point[1] * meters_per_pixel + bounds[0][2]
        # Assuming y-coordinate is 0 or some default value since it's not specified in the 2D points
        y = agent_height  # Default value for y-coordinate
        points_3d.append(np.array([x, y, z]))
    return points_3d


def stitch_images(images):
    # Ensure all images have the same height
    heights = [img.shape[0] for img in images]
    min_height = min(heights)

    # Resize images to have the same height
    resized_images = [
        cv2.resize(img, (int(img.shape[1] * min_height / img.shape[0]), min_height))
        for img in images
    ]

    # Concatenate images horizontally
    panorama = np.hstack(resized_images)

    return panorama


# Function to load an image from a URL
def load_image(url):
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    return Image.open(BytesIO(response.content)).convert("RGB")


# Function to preprocess the image for MobileNetV3
def preprocess_image(image, target_size=224):
    preprocess = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(image).unsqueeze(0)


# Function to display the original image
def display_image(image, title="Original Image"):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.title(title)
    plt.show()


def slerp(q0, q1, t):
    # Manually compute the dot product of two quaternions
    dot = (
        q0.vector.x * q1.vector.x
        + q0.vector.y * q1.vector.y
        + q0.vector.z * q1.vector.z
        + q0.scalar * q1.scalar
    )

    # Clamping the dot product to stay within the bounds of acos()
    dot = max(min(dot, 1.0), -1.0)

    # If the quaternions are nearly identical, interpolate linearly
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result.normalized()

    # Calculate the angle theta for interpolation
    theta_0 = np.arccos(dot)  # Angle between quaternions
    theta = theta_0 * t

    # Compute the sin(theta) for both
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    # Compute the scaling coefficients
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    # Compute the interpolated quaternion
    interpolated_q = (q0 * s0) + (q1 * s1)
    return interpolated_q.normalized()


# Helper function to convert degrees to radians
def degrees_to_magnum_rad(deg):
    return mn.Rad(deg * np.pi / 180.0)


# Class to hold and manage the points
class PointsManager:
    def __init__(self):
        self.points = np.array([])
        self.new_points = []
        self.random_point = None

    def generate_random_points(self, num_points, height, width):
        self.points = []
        while len(self.points) < num_points:
            row, col = np.random.randint(0, height), np.random.randint(0, width)
            self.points.append((row, col))
        return self.points

    def generate_lattice_points(self, height, width):
        hex_radius = 2
        hex_height = np.sqrt(3) * hex_radius  # Height of each hexagon
        points = []
        for row in range(0, height, int(hex_height)):
            for col in range(0, width, int(3 * hex_radius)):
                # Offset every other row by half the hex_radius
                x_offset = (row % int(2 * hex_radius)) * (hex_radius / 2)
                x = col + x_offset
                if x < width:  # Ensure x is within bounds
                    points.append((int(row), int(x)))

        self.points = np.array(points)

    def generate_grid_points(self, height, width, step=12):
        points = [
            (row, col)
            for row in range(0, height, step)
            for col in range(0, width, step)
        ]
        self.points = np.array(points)

    def generate_grid_points_centered(self, center, height, width, step=1):
            # Calculate the bounds
            min_height = int(np.round(center[0] - (height / 2)))
            max_height = int(np.round(center[0] + (height / 2)))
            min_width = int(np.round(center[1] - (width / 2)))
            max_width = int(np.round(center[1] + (width / 2)))
            
            # Generate the grid points
            points = [
                (row, col)
                for row in range(min_height, max_height, step)
                for col in range(min_width, max_width, step)
            ]
            
            # Store the points as a numpy array
            self.points = np.array(points)
            
    def generate_random_point(self, height, width):
        row, col = np.random.randint(0, height), np.random.randint(0, width)
        self.random_point = (row, col)
        return (row, col)


def extract_sift_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors


def aggregate_descriptors(images):
    all_descriptors = []
    for image in images:
        _, descriptors = extract_sift_features(image)
        if descriptors is not None:
            all_descriptors.append(descriptors)
    # Stack all descriptors into a single numpy array
    all_descriptors = np.vstack(all_descriptors)
    return all_descriptors


def train_kmeans(descriptors, n_clusters=128):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(descriptors)
    return kmeans


def display_map_path(topdown_map, key_points=None):
    plt.figure(figsize=(40, 30))  # Increased figure size
    ax = plt.subplot(1, 1, 1)
    # ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    if key_points is not None:
        for i, point in enumerate(key_points):
            if i == 0:
                plt.plot(
                    point[0],
                    point[1],
                    marker="o",
                    markersize=14,
                    alpha=0.8,
                    color="red",
                )
                plt.text(
                    point[0],
                    point[1],
                    "R",
                    color="black",
                    fontsize=14,
                    ha="center",
                    va="center",
                )
            elif i == len(key_points) - 1:
                plt.plot(
                    point[0],
                    point[1],
                    marker="o",
                    markersize=14,
                    alpha=0.8,
                    color="red",
                )
                plt.text(
                    point[0],
                    point[1],
                    "N",
                    color="black",
                    fontsize=14,
                    ha="center",
                    va="center",
                )
            else:
                plt.plot(
                    point[0],
                    point[1],
                    marker="o",
                    markersize=12,
                    alpha=0.8,
                    color="blue",
                )
                plt.text(
                    point[0],
                    point[1],
                    str(i),
                    color="white",
                    fontsize=10,
                    ha="center",
                    va="center",
                )
    plt.title(
        "Navigation of Agent from Random Point to NN in 3D Environment", fontsize=30
    )
    plt.show(block=False)




def display_graph_navigation(topdown_map, graph_coords, movement_coords):
    plt.figure(figsize=(40, 30))  # Increased figure size
    ax = plt.subplot(1, 1, 1)
    # ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map
    for i, point in enumerate(graph_coords):
            plt.plot(
                point[0],
                point[1],
                marker="o",
                markersize=25,
                alpha=0.8,
                color="yellow",
            )
            plt.text(
                point[0],
                point[1],
                str(f'N{i}'),
                color="black",
                fontsize=10,
                ha="center",
                va="center",
            )
    if movement_coords is not None:
        for i, point in enumerate(movement_coords):
            plt.plot(
                point[0],
                point[1],
                marker="o",
                markersize=12,
                alpha=0.8,
                color="blue",
            )
            plt.text(
                point[0],
                point[1],
                str(i),
                color="white",
                fontsize=10,
                ha="center",
                va="center",
            )
    plt.title(
        "Navigation of Agent using Graph", fontsize=30
    )
    plt.show(block=False)



def plot_point(point, marker, markersize, color, alpha, text, text_color, fontsize, ha, va):
    plt.plot(point[0], point[1], marker=marker, markersize=markersize, alpha=alpha, color=color)
    if text:
        plt.text(point[0], point[1], text, color=text_color, fontsize=fontsize, ha=ha, va=va)

def plot_path(path, start_label, end_label, intermediate_color, intermediate_size, start_end_size):
    for i, point in enumerate(path):
        if i == 0:
            plot_point(point, "o", start_end_size, "red", 0.8, start_label, "black", 14, "center", "center")
        elif i == len(path) - 1:
            plot_point(point, "o", start_end_size, "red", 0.8, end_label, "black", 14, "center", "center")
        else:
            plot_point(point, "o", intermediate_size, intermediate_color, 0.8, str(i), "white", 10, "center", "center")

def display_full_process(topdown_map, path1, path2, path3):
    plt.figure(figsize=(40, 30))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    
    plot_path(path1, "R2", "N2", "blue", 12, 14)
    plot_path(path2[1:len(path1)-1], None, None, "green", 14, 14)
    plot_path(path3, "N1", "R1", "blue", 12, 14)
    
    plt.title("Navigation of Agent from Random Point to NN in 3D Environment", fontsize=30)
    plt.show(block=False)

