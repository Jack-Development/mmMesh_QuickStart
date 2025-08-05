import numpy as np
import os
from tqdm import tqdm
import configuration as cfg
import matplotlib.pyplot as plt
import cv2


def create_point_cloud_frame(points, title, elev=20, azim=45):
    """
    Create a single frame visualization of a point cloud.

    Args:
        points: numpy array of shape (N, 6) where columns are [x, y, z, velocity, energy, range]
        title: Title for the plot
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view

    Returns:
        numpy array: Frame as BGR image for video
    """
    fig = plt.figure(figsize=(12, 8))

    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection="3d")

    if points.shape[0] > 0 and np.any(points[:, :3] != 0):
        # Color points by velocity for better visualization
        scatter = ax1.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 3],
            cmap="viridis",
            s=20,
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax1, label="Velocity (m/s)", shrink=0.8)
    else:
        # Empty or zero points - show as gray dots at origin
        ax1.scatter([0], [0], [0], c="gray", s=50, alpha=0.5)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title(f"{title}\n3D Point Cloud ({points.shape[0]} points)")
    ax1.view_init(elev=elev, azim=azim)

    # Set expanded axis limits to accommodate mmWave data better
    # Based on typical mmWave detection ranges and radar location
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-1, 5])
    ax1.set_zlim([-0.5, 2])

    # 2D projections
    # XY plane (top view)
    ax2 = fig.add_subplot(222)
    if points.shape[0] > 0 and np.any(points[:, :3] != 0):
        scatter2 = ax2.scatter(
            points[:, 0], points[:, 1], c=points[:, 3], cmap="viridis", s=20, alpha=0.7
        )
        plt.colorbar(scatter2, ax=ax2, label="Velocity (m/s)")
    else:
        ax2.scatter([0], [0], c="gray", s=50, alpha=0.5)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Top View (XY)")
    ax2.set_xlim([-3, 3])
    ax2.set_ylim([-1, 5])
    ax2.grid(True, alpha=0.3)

    # Velocity distribution
    ax3 = fig.add_subplot(223)
    if points.shape[0] > 0:
        ax3.hist(points[:, 3], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    ax3.set_xlabel("Velocity (m/s)")
    ax3.set_ylabel("Count")
    ax3.set_title("Velocity Distribution")
    ax3.grid(True, alpha=0.3)

    # Energy distribution
    ax4 = fig.add_subplot(224)
    if points.shape[0] > 0:
        ax4.hist(points[:, 4], bins=20, alpha=0.7, color="orange", edgecolor="black")
    ax4.set_xlabel("Energy (dB)")
    ax4.set_ylabel("Count")
    ax4.set_title("Energy Distribution")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    plt.close(fig)
    return frame


def create_adaptive_bounds(train_data, test_data):
    """
    Calculate adaptive axis bounds based on actual data distribution from both datasets.

    Args:
        train_data: Training data array of shape (N_train_subjects, T_train, P, 6)
        test_data: Test data array of shape (N_test_subjects, T_test, P, 6)

    Returns:
        dict: Dictionary with axis bounds
    """
    all_points = []

    # Process train data
    train_flat = train_data.reshape(-1, train_data.shape[-1])
    train_valid_mask = np.any(train_flat[:, :3] != 0, axis=1)
    train_valid_points = train_flat[train_valid_mask]
    if len(train_valid_points) > 0:
        all_points.append(train_valid_points)

    # Process test data
    test_flat = test_data.reshape(-1, test_data.shape[-1])
    test_valid_mask = np.any(test_flat[:, :3] != 0, axis=1)
    test_valid_points = test_flat[test_valid_mask]
    if len(test_valid_points) > 0:
        all_points.append(test_valid_points)

    if not all_points:
        # Fallback bounds if no valid points
        return {"x_lim": [-3, 3], "y_lim": [-1, 5], "z_lim": [-0.5, 2]}

    # Combine all valid points
    combined_points = np.concatenate(all_points, axis=0)

    # Calculate percentile-based bounds (removes outliers)
    x_bounds = np.percentile(combined_points[:, 0], [2, 98])
    y_bounds = np.percentile(combined_points[:, 1], [2, 98])
    z_bounds = np.percentile(combined_points[:, 2], [2, 98])

    # Add some padding
    x_padding = (x_bounds[1] - x_bounds[0]) * 0.1
    y_padding = (y_bounds[1] - y_bounds[0]) * 0.1
    z_padding = (z_bounds[1] - z_bounds[0]) * 0.1

    return {
        "x_lim": [x_bounds[0] - x_padding, x_bounds[1] + x_padding],
        "y_lim": [y_bounds[0] - y_padding, y_bounds[1] + y_padding],
        "z_lim": [z_bounds[0] - z_padding, z_bounds[1] + z_padding],
    }


def create_subject_video(
    subject_data,
    subject_idx,
    split_name,
    output_dir,
    fps=10,
    bounds=None,
    max_frames=None,
):
    """
    Create a video for a single subject showing all frames as a sequence.

    Args:
        subject_data: Data for one subject of shape (T, P, 6)
        subject_idx: Index of the subject in the dataset
        split_name: "train" or "test"
        output_dir: Directory to save the video
        fps: Frames per second for the video
        bounds: Dictionary with axis bounds, if None will use defaults
        max_frames: Maximum number of frames to include in video (None for all frames)
    """
    video_filename = f"{split_name}_subject_{subject_idx:03d}.mp4"
    video_path = os.path.join(output_dir, video_filename)

    # Limit frames if max_frames is specified
    total_frames = subject_data.shape[0]
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
        subject_data = subject_data[:total_frames]

    # Get video dimensions from first frame
    sample_frame = create_point_cloud_frame_with_bounds(
        subject_data[0], f"Subject {subject_idx} - {split_name.title()} Frame 0", bounds
    )
    height, width, _ = sample_frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {video_path}")
        return None

    # Create frames for the video
    for frame_idx in tqdm(
        range(total_frames),
        desc=f"{split_name.title()} Subject {subject_idx}",
        leave=False,
    ):
        points = subject_data[frame_idx]  # Shape: (P, 6)
        title = f"Subject {subject_idx} - {split_name.title()} Frame {frame_idx}"

        frame = create_point_cloud_frame_with_bounds(points, title, bounds)
        video_writer.write(frame)

    video_writer.release()
    return video_path


def create_point_cloud_frame_with_bounds(points, title, bounds=None, elev=20, azim=45):
    """
    Create a single frame visualization with custom bounds.
    """
    if bounds is None:
        bounds = {"x_lim": [-3, 3], "y_lim": [-1, 5], "z_lim": [-0.5, 2]}

    fig = plt.figure(figsize=(12, 8))

    # 3D scatter plot
    ax1 = fig.add_subplot(221, projection="3d")

    if points.shape[0] > 0 and np.any(points[:, :3] != 0):
        # Color points by velocity for better visualization
        scatter = ax1.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=points[:, 3],
            cmap="viridis",
            s=20,
            alpha=0.7,
        )
        plt.colorbar(scatter, ax=ax1, label="Velocity (m/s)", shrink=0.8)
    else:
        # Empty or zero points - show as gray dots at origin
        ax1.scatter([0], [0], [0], c="gray", s=50, alpha=0.5)

    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")
    ax1.set_title(f"{title}\n3D Point Cloud ({points.shape[0]} points)")
    ax1.view_init(elev=elev, azim=azim)

    # Use adaptive bounds
    ax1.set_xlim(bounds["x_lim"])
    ax1.set_ylim(bounds["y_lim"])
    ax1.set_zlim(bounds["z_lim"])

    # 2D projections
    # XY plane (top view)
    ax2 = fig.add_subplot(222)
    if points.shape[0] > 0 and np.any(points[:, :3] != 0):
        scatter2 = ax2.scatter(
            points[:, 0], points[:, 1], c=points[:, 3], cmap="viridis", s=20, alpha=0.7
        )
        plt.colorbar(scatter2, ax=ax2, label="Velocity (m/s)")
    else:
        ax2.scatter([0], [0], c="gray", s=50, alpha=0.5)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Top View (XY)")
    ax2.set_xlim(bounds["x_lim"])
    ax2.set_ylim(bounds["y_lim"])
    ax2.grid(True, alpha=0.3)

    # Velocity distribution
    ax3 = fig.add_subplot(223)
    if points.shape[0] > 0:
        ax3.hist(points[:, 3], bins=20, alpha=0.7, color="skyblue", edgecolor="black")
    ax3.set_xlabel("Velocity (m/s)")
    ax3.set_ylabel("Count")
    ax3.set_title("Velocity Distribution")
    ax3.grid(True, alpha=0.3)

    # Energy distribution
    ax4 = fig.add_subplot(224)
    if points.shape[0] > 0:
        ax4.hist(points[:, 4], bins=20, alpha=0.7, color="orange", edgecolor="black")
    ax4.set_xlabel("Energy (dB)")
    ax4.set_ylabel("Count")
    ax4.set_title("Energy Distribution")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)

    plt.close(fig)
    return frame


def create_dataset_visualizations(
    train_data, test_data, output_dir, fps=10, max_frames=None
):
    """
    Create video visualizations for train and test datasets.

    Args:
        train_data: Training data array of shape (N_subjects, T, P, 6)
        test_data: Test data array of shape (N_subjects, T, P, 6)
        output_dir: Directory to save visualizations
        fps: Frames per second for videos
        max_frames: Maximum number of frames per video (None for all frames)
    """
    vis_dir = os.path.join(output_dir, "visualizations")
    train_vis_dir = os.path.join(vis_dir, "train_videos")
    test_vis_dir = os.path.join(vis_dir, "test_videos")

    # Create directories
    os.makedirs(train_vis_dir, exist_ok=True)
    os.makedirs(test_vis_dir, exist_ok=True)

    print(f"\nCreating video visualizations in {vis_dir}")
    print(f"Video settings: {fps} FPS")
    if max_frames is not None:
        print(f"Frame limit: {max_frames} frames per video")
    else:
        print("Frame limit: No limit (all frames)")
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    # Calculate adaptive bounds based on both datasets separately
    print("Calculating optimal visualization bounds...")
    bounds = create_adaptive_bounds(train_data, test_data)

    print(f"Using bounds: X{bounds['x_lim']}, Y{bounds['y_lim']}, Z{bounds['z_lim']}")

    # Create training videos
    print("Creating training videos...")
    train_video_paths = []
    for subject_idx in tqdm(range(train_data.shape[0]), desc="Train subjects"):
        video_path = create_subject_video(
            train_data[subject_idx],
            subject_idx,
            "train",
            train_vis_dir,
            fps,
            bounds,
            max_frames,
        )
        if video_path:
            train_video_paths.append(video_path)

    # Create test videos
    print("Creating test videos...")
    test_video_paths = []
    for subject_idx in tqdm(range(test_data.shape[0]), desc="Test subjects"):
        video_path = create_subject_video(
            test_data[subject_idx],
            subject_idx,
            "test",
            test_vis_dir,
            fps,
            bounds,
            max_frames,
        )
        if video_path:
            test_video_paths.append(video_path)

    # Create dataset summary visualization (still as image)
    create_dataset_summary(train_data, test_data, vis_dir, fps, max_frames)

    print("\nVideo visualizations completed!")
    print(f"  Training videos ({len(train_video_paths)}): {train_vis_dir}")
    print(f"  Test videos ({len(test_video_paths)}): {test_vis_dir}")
    if max_frames is not None:
        print(f"  Each video limited to {max_frames} frames")
    print(
        "  Videos are named by subject index: train_subject_XXX.mp4 / test_subject_XXX.mp4"
    )


def create_dataset_summary(train_data, test_data, vis_dir, fps=10, max_frames=None):
    """
    Create summary visualizations comparing train and test datasets.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Dataset Summary: Train vs Test Comparison", fontsize=16)

    # Calculate statistics
    train_point_counts = []
    test_point_counts = []
    train_velocities = []
    test_velocities = []
    train_energies = []
    test_energies = []

    # Limit frames for analysis if specified
    train_frames_to_analyze = train_data.shape[1]
    test_frames_to_analyze = test_data.shape[1]
    if max_frames is not None:
        train_frames_to_analyze = min(train_frames_to_analyze, max_frames)
        test_frames_to_analyze = min(test_frames_to_analyze, max_frames)

    for subject_idx in range(train_data.shape[0]):
        for frame_idx in range(train_frames_to_analyze):
            points = train_data[subject_idx, frame_idx]
            non_zero_mask = np.any(points[:, :3] != 0, axis=1)
            valid_points = points[non_zero_mask]

            train_point_counts.append(len(valid_points))
            if len(valid_points) > 0:
                train_velocities.extend(valid_points[:, 3])
                train_energies.extend(valid_points[:, 4])

    for subject_idx in range(test_data.shape[0]):
        for frame_idx in range(test_frames_to_analyze):
            points = test_data[subject_idx, frame_idx]
            non_zero_mask = np.any(points[:, :3] != 0, axis=1)
            valid_points = points[non_zero_mask]

            test_point_counts.append(len(valid_points))
            if len(valid_points) > 0:
                test_velocities.extend(valid_points[:, 3])
                test_energies.extend(valid_points[:, 4])

    # Plot 1: Point count distribution
    axes[0, 0].hist(train_point_counts, bins=30, alpha=0.7, label="Train", density=True)
    axes[0, 0].hist(test_point_counts, bins=30, alpha=0.7, label="Test", density=True)
    axes[0, 0].set_xlabel("Points per Frame")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_title("Point Count Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Velocity distribution
    if train_velocities and test_velocities:
        axes[0, 1].hist(
            train_velocities, bins=50, alpha=0.7, label="Train", density=True
        )
        axes[0, 1].hist(test_velocities, bins=50, alpha=0.7, label="Test", density=True)
    axes[0, 1].set_xlabel("Velocity (m/s)")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Velocity Distribution")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Energy distribution
    if train_energies and test_energies:
        axes[0, 2].hist(train_energies, bins=50, alpha=0.7, label="Train", density=True)
        axes[0, 2].hist(test_energies, bins=50, alpha=0.7, label="Test", density=True)
    axes[0, 2].set_xlabel("Energy (dB)")
    axes[0, 2].set_ylabel("Density")
    axes[0, 2].set_title("Energy Distribution")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Dataset shape comparison
    axes[1, 0].bar(
        ["Train Subjects", "Test Subjects"],
        [train_data.shape[0], test_data.shape[0]],
        color=["blue", "orange"],
        alpha=0.7,
    )
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Number of Subjects")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 5: Frame count comparison
    actual_train_frames = train_frames_to_analyze
    actual_test_frames = test_frames_to_analyze

    axes[1, 1].bar(
        ["Train Frames", "Test Frames"],
        [actual_train_frames, actual_test_frames],
        color=["blue", "orange"],
        alpha=0.7,
    )
    axes[1, 1].set_ylabel("Frames per Subject")
    if max_frames is not None:
        axes[1, 1].set_title(f"Frames per Subject (Limited to {max_frames})")
    else:
        axes[1, 1].set_title("Frames per Subject")
    axes[1, 1].grid(True, alpha=0.3)

    # Plot 6: Summary statistics
    axes[1, 2].axis("off")
    summary_text = f"""
Dataset Statistics:

Train Data:
  Subjects: {train_data.shape[0]}
  Frames per subject: {train_data.shape[1]}
  Points per frame: {train_data.shape[2]}
  Avg points/frame: {np.mean(train_point_counts):.1f}

Test Data:
  Subjects: {test_data.shape[0]}
  Frames per subject: {test_data.shape[1]}
  Points per frame: {test_data.shape[2]}
  Avg points/frame: {np.mean(test_point_counts):.1f}

Configuration:
  PC_SIZE: {cfg.PC_SIZE}
  Split method: {cfg.SPLIT_METHOD}
  Test ratio: {cfg.TEST_RATIO}
  Video FPS: {fps}
  Max frames: {max_frames if max_frames else 'No limit'}
    """

    axes[1, 2].text(
        0.1,
        0.9,
        summary_text,
        transform=axes[1, 2].transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
    )

    plt.tight_layout()
    summary_path = os.path.join(vis_dir, "dataset_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Dataset summary saved: {summary_path}")
