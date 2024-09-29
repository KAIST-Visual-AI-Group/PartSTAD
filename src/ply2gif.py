import open3d as o3d
import numpy as np
import imageio
import matplotlib.pyplot as plt
import os

# Normalize the point cloud to fit within a unit sphere
def normalize_point_cloud(points):
    center = points.mean(axis=0)
    points -= center  # Center the points
    scale = np.linalg.norm(points, axis=1).max()  # Find the farthest point
    points /= scale  # Normalize so that the object fits within a unit sphere
    return points

# Rotate the point cloud 90 degrees along the x-axis
def rotate_point_cloud_x(points):
    rotation_matrix = np.array([[1, 0, 0],
                                [0, np.cos(np.pi/2), -np.sin(np.pi/2)],
                                [0, np.sin(np.pi/2), np.cos(np.pi/2)]])
    return points @ rotation_matrix.T  # Apply rotation to the points

# Load the point cloud from a PLY file and normalize it
def load_and_normalize_point_cloud(file_path):
    point_cloud = o3d.io.read_point_cloud(file_path)
    
    points = np.asarray(point_cloud.points)
    colors = None

    # Check if the point cloud has color information
    if point_cloud.has_colors():
        colors = np.asarray(point_cloud.colors)
        print("Point cloud has color information.")
    else:
        print("Point cloud does not have color information.")

    # Check if points were correctly loaded
    if points.size == 0:
        raise ValueError("The point cloud appears to be empty or incorrectly loaded.")

    print(f"Loaded point cloud shape: {points.shape}")
    
    # Normalize the point cloud
    normalized_points = normalize_point_cloud(points)
    
    # Rotate the point cloud 90 degrees along the x-axis
    rotated_points = rotate_point_cloud_x(normalized_points)
    
    return rotated_points, colors

# Create a rotating gif of the normalized point cloud with a fixed camera radius
def create_rotation_gif(points, colors, gif_filename="pointcloud_rotation.gif", n_frames=18, camera_radius=1.2, elevation_angle=30):
    images = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # The center of the normalized point cloud is already at (0, 0, 0)

    # Generate views by rotating the camera around the normalized point cloud
    for i in range(n_frames):
        ax.clear()

        # Ensure points is a 2D array with shape (N, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("The points array must be 2D with shape (N, 3).")

        # Calculate the camera position on a circular path around the normalized object
        angle_rad = np.radians(360 / n_frames * i)  # Calculate the angle in radians for this frame
        cam_x = camera_radius * np.cos(angle_rad)  # X coordinate of the camera
        cam_y = camera_radius * np.sin(angle_rad)  # Y coordinate of the camera
        cam_z = camera_radius * np.sin(np.radians(elevation_angle))  # Z coordinate of the camera (elevation)

        # Plot the point cloud with or without color
        if colors is not None:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)

        # Set the camera position and focus
        ax.view_init(elev=elevation_angle, azim=np.degrees(angle_rad))  # Adjust elevation and azimuth for rotation
        ax.set_xlim([-camera_radius, camera_radius])  # Fixed limits based on the normalized point cloud
        ax.set_ylim([-camera_radius, camera_radius])
        ax.set_zlim([-camera_radius, camera_radius])

        # Remove grid, ticks, and bounding box
        ax.grid(False)  # Remove the grid
        ax.set_xticks([])  # Remove X-axis ticks
        ax.set_yticks([])  # Remove Y-axis ticks
        ax.set_zticks([])  # Remove Z-axis ticks
        ax.set_axis_off()  # Remove the axis box and background

        # Capture the plot as an image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

    plt.close()

    # Save the images as a GIF with infinite looping
    imageio.mimsave(gif_filename, images, fps=10, loop=0)  # `loop=0` for infinite looping
    print(f"GIF saved as {gif_filename}")

# Example usage
if __name__=='__main__':
    ply_file_path = "/home/dld/git_misc/partstad/PartSTAD/demo_examples_result/test_result/Chair/40067/semantic_seg/all.ply"  # Replace with your PLY file path
    points, colors = load_and_normalize_point_cloud(ply_file_path)

    # Create a rotating gif
    create_rotation_gif(points, colors, gif_filename="pointcloud_rotation.gif")
