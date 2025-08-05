import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import tempfile


class TrainingVisualizer:
    """
    Helper class to visualize mmWave point clouds, ground truth skeletons,
    and predicted skeletons during training.
    """

    def __init__(self, output_dir="./visualization"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectories for different types of visualizations
        self.train_dir = os.path.join(output_dir, "training")
        self.eval_dir = os.path.join(output_dir, "evaluation")
        self.video_dir = os.path.join(output_dir, "videos")

        for d in [self.train_dir, self.eval_dir, self.video_dir]:
            os.makedirs(d, exist_ok=True)

        # SMPL skeleton connections for drawing bones
        self.skeleton_connections = [
            (0, 1),
            (0, 2),
            (0, 3),  # pelvis to legs and spine
            (1, 4),
            (2, 5),  # upper legs
            (4, 7),
            (5, 8),  # lower legs
            (3, 6),  # spine
            (6, 9),
            (9, 12),
            (12, 15),  # spine to head
            (9, 13),
            (9, 14),  # shoulders
            (13, 16),
            (14, 17),  # upper arms
            (16, 18),
            (17, 19),  # lower arms
            (18, 20),
            (19, 21),  # hands
            (20, 22),
            (21, 23),  # fingers
        ]

    def plot_skeleton_comparison(
        self,
        pc_data,
        gt_skeleton,
        pred_skeleton,
        step,
        batch_idx=0,
        frame_idx=0,
        save=True,
        output_dir=None,
    ):
        """
        Plot comparison of mmWave points, ground truth skeleton, and predicted skeleton.
        """

        # Convert tensors to numpy if needed
        if torch.is_tensor(pc_data):
            pc_data = pc_data.detach().cpu().numpy()
        if torch.is_tensor(gt_skeleton):
            gt_skeleton = gt_skeleton.detach().cpu().numpy()
        if torch.is_tensor(pred_skeleton):
            pred_skeleton = pred_skeleton.detach().cpu().numpy()

        # Extract single frame data
        pc_frame = pc_data[batch_idx, frame_idx]  # [P, 6]
        gt_frame = gt_skeleton[batch_idx, frame_idx]  # [24, 3]
        pred_frame = pred_skeleton[batch_idx, frame_idx]  # [24, 3]

        # Create figure with 3 subplots
        fig = plt.figure(figsize=(15, 5))

        # Plot 1: mmWave point cloud
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.scatter(
            pc_frame[:, 0], pc_frame[:, 1], pc_frame[:, 2], s=10, c="blue", alpha=0.6
        )
        ax1.set_title("mmWave Point Cloud")
        self._set_axis_properties(ax1)

        # Plot 2: Ground truth skeleton
        ax2 = fig.add_subplot(132, projection="3d")
        self._plot_skeleton(ax2, gt_frame, color="green", label="Ground Truth")
        ax2.set_title("Ground Truth Skeleton")
        self._set_axis_properties(ax2)

        # Plot 3: Predicted skeleton
        ax3 = fig.add_subplot(133, projection="3d")
        self._plot_skeleton(ax3, pred_frame, color="red", label="Predicted")
        ax3.set_title("Predicted Skeleton")
        self._set_axis_properties(ax3)

        plt.tight_layout()

        if save:
            save_dir = output_dir if output_dir else self.train_dir
            filename = f"step_{step:06d}_batch_{batch_idx}_frame_{frame_idx}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            return filepath
        else:
            plt.show()
            return None

    def plot_skeleton_overlay(
        self,
        pc_data,
        gt_skeleton,
        pred_skeleton,
        step,
        batch_idx=0,
        frame_idx=0,
        save=True,
        output_dir=None,
    ):
        """
        Plot overlay comparison with GT and predicted skeletons in same plot.
        """

        # Convert tensors to numpy if needed
        if torch.is_tensor(pc_data):
            pc_data = pc_data.detach().cpu().numpy()
        if torch.is_tensor(gt_skeleton):
            gt_skeleton = gt_skeleton.detach().cpu().numpy()
        if torch.is_tensor(pred_skeleton):
            pred_skeleton = pred_skeleton.detach().cpu().numpy()

        # Extract single frame data
        pc_frame = pc_data[batch_idx, frame_idx]
        gt_frame = gt_skeleton[batch_idx, frame_idx]
        pred_frame = pred_skeleton[batch_idx, frame_idx]

        # Create figure with 2 subplots
        fig = plt.figure(figsize=(12, 5))

        # Plot 1: mmWave point cloud
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.scatter(
            pc_frame[:, 0],
            pc_frame[:, 1],
            pc_frame[:, 2],
            s=10,
            c="blue",
            alpha=0.6,
            label="mmWave Points",
        )
        ax1.set_title("mmWave Point Cloud")
        ax1.legend()
        self._set_axis_properties(ax1)

        # Plot 2: Skeleton overlay
        ax2 = fig.add_subplot(122, projection="3d")
        self._plot_skeleton(
            ax2, gt_frame, color="green", label="Ground Truth", alpha=0.8
        )
        self._plot_skeleton(ax2, pred_frame, color="red", label="Predicted", alpha=0.8)
        ax2.set_title("Skeleton Comparison")
        ax2.legend()
        self._set_axis_properties(ax2)

        plt.tight_layout()

        if save:
            save_dir = output_dir if output_dir else self.train_dir
            filename = (
                f"overlay_step_{step:06d}_batch_{batch_idx}_frame_{frame_idx}.png"
            )
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close()
            return filepath
        else:
            plt.show()
            return None

    def create_batch_video(
        self,
        pc_data,
        gt_skeleton,
        pred_skeleton,
        step,
        batch_idx=0,
        fps=10,
        video_type="training",
    ):
        """
        Create a video showing all frames from a sequence for a specific batch sample.

        Args:
            pc_data: Point cloud data [B, T, P, 6]
            gt_skeleton: Ground truth skeleton [B, T, 24, 3]
            pred_skeleton: Predicted skeleton [B, T, 24, 3]
            step: Training step number
            batch_idx: Which sample from batch to visualize
            fps: Frames per second for video
            video_type: "training" or "evaluation"
        """

        # Convert tensors to numpy if needed
        if torch.is_tensor(pc_data):
            pc_data = pc_data.detach().cpu().numpy()
        if torch.is_tensor(gt_skeleton):
            gt_skeleton = gt_skeleton.detach().cpu().numpy()
        if torch.is_tensor(pred_skeleton):
            pred_skeleton = pred_skeleton.detach().cpu().numpy()

        seq_length = pc_data.shape[1]

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_files = []

            # Generate all frames
            for frame_idx in range(seq_length):
                frame_path = self.plot_skeleton_overlay(
                    pc_data,
                    gt_skeleton,
                    pred_skeleton,
                    step,
                    batch_idx,
                    frame_idx,
                    save=True,
                    output_dir=temp_dir,
                )
                frame_files.append(frame_path)

            # Create video from frames
            video_filename = f"{video_type}_step_{step:06d}_batch_{batch_idx}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)

            self._create_video_from_frames(frame_files, video_path, fps)

            return video_path

    def create_full_batch_video(
        self,
        pc_data,
        gt_skeleton,
        pred_skeleton,
        step,
        max_samples=4,
        fps=10,
        video_type="training",
    ):
        """
        Create a video showing multiple samples from a batch in a grid layout.

        Args:
            pc_data: Point cloud data [B, T, P, 6]
            gt_skeleton: Ground truth skeleton [B, T, 24, 3]
            pred_skeleton: Predicted skeleton [B, T, 24, 3]
            step: Training step number
            max_samples: Maximum number of batch samples to show
            fps: Frames per second for video
            video_type: "training" or "evaluation"
        """

        # Convert tensors to numpy if needed
        if torch.is_tensor(pc_data):
            pc_data = pc_data.detach().cpu().numpy()
        if torch.is_tensor(gt_skeleton):
            gt_skeleton = gt_skeleton.detach().cpu().numpy()
        if torch.is_tensor(pred_skeleton):
            pred_skeleton = pred_skeleton.detach().cpu().numpy()

        batch_size = min(pc_data.shape[0], max_samples)
        seq_length = pc_data.shape[1]

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_files = []

            # Generate frames for the entire sequence
            for frame_idx in range(seq_length):
                frame_path = self._create_grid_frame(
                    pc_data,
                    gt_skeleton,
                    pred_skeleton,
                    step,
                    frame_idx,
                    batch_size,
                    temp_dir,
                )
                frame_files.append(frame_path)

            # Create video from frames
            video_filename = f"{video_type}_grid_step_{step:06d}.mp4"
            video_path = os.path.join(self.video_dir, video_filename)

            self._create_video_from_frames(frame_files, video_path, fps)

            return video_path

    def _create_grid_frame(
        self,
        pc_data,
        gt_skeleton,
        pred_skeleton,
        step,
        frame_idx,
        batch_size,
        output_dir,
    ):
        """Create a single frame showing multiple batch samples in a grid."""

        # Calculate grid dimensions
        grid_cols = min(batch_size, 2)  # Max 2 columns
        grid_rows = (batch_size + grid_cols - 1) // grid_cols

        fig = plt.figure(figsize=(12 * grid_cols, 4 * grid_rows))
        fig.suptitle(f"Training Step {step} - Frame {frame_idx}", fontsize=16)

        for batch_idx in range(batch_size):
            # Extract data for this batch sample and frame
            pc_frame = pc_data[batch_idx, frame_idx]
            gt_frame = gt_skeleton[batch_idx, frame_idx]
            pred_frame = pred_skeleton[batch_idx, frame_idx]

            # Calculate subplot positions (3 plots per batch sample)
            row = batch_idx // grid_cols
            col = batch_idx % grid_cols

            # Adjust subplot indexing for grid layout
            subplot_base = row * (grid_cols * 3) + col * 3 + 1

            # mmWave point cloud
            ax1 = fig.add_subplot(
                grid_rows, grid_cols * 3, subplot_base, projection="3d"
            )
            ax1.scatter(
                pc_frame[:, 0], pc_frame[:, 1], pc_frame[:, 2], s=5, c="blue", alpha=0.6
            )
            ax1.set_title(f"Sample {batch_idx} - mmWave")
            self._set_axis_properties(ax1, small=True)

            # Ground truth skeleton
            ax2 = fig.add_subplot(
                grid_rows, grid_cols * 3, subplot_base + 1, projection="3d"
            )
            self._plot_skeleton(ax2, gt_frame, color="green", alpha=0.8)
            ax2.set_title(f"Sample {batch_idx} - GT")
            self._set_axis_properties(ax2, small=True)

            # Predicted skeleton
            ax3 = fig.add_subplot(
                grid_rows, grid_cols * 3, subplot_base + 2, projection="3d"
            )
            self._plot_skeleton(ax3, pred_frame, color="red", alpha=0.8)
            ax3.set_title(f"Sample {batch_idx} - Pred")
            self._set_axis_properties(ax3, small=True)

        plt.tight_layout()

        filename = f"grid_frame_{frame_idx:04d}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_video_from_frames(self, frame_files, output_path, fps=10):
        """Create MP4 video from a list of frame files."""

        if not frame_files:
            print("No frames to create video")
            return

        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_files[0])
        if first_frame is None:
            print(f"Could not read first frame: {frame_files[0]}")
            return

        height, width, _ = first_frame.shape

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not video_writer.isOpened():
            print(f"Could not open video writer for {output_path}")
            return

        # Write frames to video
        for frame_file in frame_files:
            frame = cv2.imread(frame_file)
            if frame is not None:
                video_writer.write(frame)
            else:
                print(f"Could not read frame: {frame_file}")

        video_writer.release()
        print(f"Created video: {output_path}")

    def visualize_evaluation_data(self, trainer, step, save_every_n_steps=1000):
        """
        Visualize evaluation/test data.
        """
        if step % save_every_n_steps != 0:
            return None

        try:
            # Get test data
            pc_test = trainer.dataset.get_test()  # [N, T, P, 6]

            # Use first sample for visualization
            sample_idx = 0
            pc_sample = pc_test[sample_idx : sample_idx + 1]  # [1, T, P, 6]

            # Get corresponding mocap data
            pquat_sample = trainer.dataset.m_pquat_test[
                sample_idx : sample_idx + 1
            ]  # [1, T, J, 3, 3]
            trans_sample = trainer.dataset.m_trans_test[
                sample_idx : sample_idx + 1
            ]  # [1, T, 3]

            # Create betas and gender for this sample
            test_length = trainer.dataset.test_length
            betas_sample = np.repeat(
                trainer.dataset.betas[sample_idx][None, None, :], test_length, axis=1
            )  # [1, T, 10]
            gender_sample = np.repeat(
                [[trainer.dataset.gender[sample_idx]]], test_length, axis=1
            )  # [1, T, 1]

            # Convert to tensors
            pc_tensor = torch.tensor(
                pc_sample, dtype=torch.float32, device=trainer.device
            )
            pquat_tensor = torch.tensor(
                pquat_sample, dtype=torch.float32, device=trainer.device
            )
            trans_tensor = torch.tensor(
                trans_sample, dtype=torch.float32, device=trainer.device
            )
            betas_tensor = torch.tensor(
                betas_sample, dtype=torch.float32, device=trainer.device
            )
            gender_tensor = torch.tensor(
                gender_sample, dtype=torch.float32, device=trainer.device
            )

            # Generate ground truth
            gt_vertices, gt_skeleton = trainer.cal_vs_from_qtbg(
                pquat_tensor, trans_tensor, betas_tensor, gender_tensor, 1, test_length
            )

            # Get model prediction
            trainer.model.eval()
            with torch.no_grad():
                batch_size = 1
                h0_g = torch.zeros(
                    (3, batch_size, 64), dtype=torch.float32, device=trainer.device
                )
                c0_g = torch.zeros(
                    (3, batch_size, 64), dtype=torch.float32, device=trainer.device
                )
                h0_a = torch.zeros(
                    (3, batch_size, 64), dtype=torch.float32, device=trainer.device
                )
                c0_a = torch.zeros(
                    (3, batch_size, 64), dtype=torch.float32, device=trainer.device
                )

                # For evaluation, we don't pass gender to the model
                pred_outputs = trainer.model(
                    pc_tensor, None, h0_g, c0_g, h0_a, c0_a, trainer.dataset.joint_size
                )
                pred_skeleton = pred_outputs[3]  # pred_s is the 4th output

            trainer.model.train()

            # Create single frame visualization
            frame_path = self.plot_skeleton_overlay(
                pc_tensor,
                gt_skeleton,
                pred_skeleton,
                step,
                batch_idx=0,
                frame_idx=0,
                save=True,
                output_dir=self.eval_dir,
            )

            # Create video for evaluation
            video_path = self.create_batch_video(
                pc_tensor,
                gt_skeleton,
                pred_skeleton,
                step,
                batch_idx=0,
                fps=10,
                video_type="evaluation",
            )

            print(f"Saved evaluation visualization: {frame_path}")
            print(f"Saved evaluation video: {video_path}")

            return frame_path, video_path

        except Exception as e:
            print(f"Error during evaluation visualization: {e}")
            return None

    def visualize_batch_sample(self, trainer, step, save_every_n_steps=1000):
        """
        Enhanced batch visualization with video creation.
        """
        if step % save_every_n_steps != 0:
            return None

        try:
            # Get a batch of data
            pc, pquat, trans, betas, gender = trainer.dataset.next_batch()

            # Convert to tensors
            pc_tensor = torch.tensor(pc, dtype=torch.float32, device=trainer.device)
            pquat_tensor = torch.tensor(
                pquat, dtype=torch.float32, device=trainer.device
            )
            trans_tensor = torch.tensor(
                trans, dtype=torch.float32, device=trainer.device
            )
            betas_tensor = torch.tensor(
                betas, dtype=torch.float32, device=trainer.device
            )
            gender_tensor = torch.tensor(
                gender, dtype=torch.float32, device=trainer.device
            )

            # Generate ground truth
            gt_vertices, gt_skeleton = trainer.cal_vs_from_qtbg(
                pquat_tensor,
                trans_tensor,
                betas_tensor,
                gender_tensor,
                trainer.batch_size,
                trainer.train_length,
            )

            # Get model prediction
            trainer.model.eval()
            with torch.no_grad():
                h0_g = torch.zeros(
                    (3, trainer.batch_size, 64),
                    dtype=torch.float32,
                    device=trainer.device,
                )
                c0_g = torch.zeros(
                    (3, trainer.batch_size, 64),
                    dtype=torch.float32,
                    device=trainer.device,
                )
                h0_a = torch.zeros(
                    (3, trainer.batch_size, 64),
                    dtype=torch.float32,
                    device=trainer.device,
                )
                c0_a = torch.zeros(
                    (3, trainer.batch_size, 64),
                    dtype=torch.float32,
                    device=trainer.device,
                )

                pred_outputs = trainer.model(
                    pc_tensor,
                    gender_tensor,
                    h0_g,
                    c0_g,
                    h0_a,
                    c0_a,
                    trainer.dataset.joint_size,
                )
                pred_skeleton = pred_outputs[3]  # pred_s is the 4th output

            trainer.model.train()

            # Create single frame visualization
            frame_path = self.plot_skeleton_overlay(
                pc_tensor,
                gt_skeleton,
                pred_skeleton,
                step,
                batch_idx=0,
                frame_idx=0,
                save=True,
                output_dir=self.train_dir,
            )

            # Create individual sample video
            single_video_path = self.create_batch_video(
                pc_tensor,
                gt_skeleton,
                pred_skeleton,
                step,
                batch_idx=0,
                fps=10,
                video_type="training",
            )

            # Create full batch grid video
            grid_video_path = self.create_full_batch_video(
                pc_tensor,
                gt_skeleton,
                pred_skeleton,
                step,
                max_samples=4,
                fps=10,
                video_type="training",
            )

            print(f"Saved training visualization: {frame_path}")
            print(f"Saved single sample video: {single_video_path}")
            print(f"Saved batch grid video: {grid_video_path}")

            return frame_path, single_video_path, grid_video_path

        except Exception as e:
            print(f"Error during training visualization: {e}")
            return None

    def _plot_skeleton(self, ax, skeleton_points, color="red", label=None, alpha=1.0):
        """
        Plot skeleton joints and connections.
        """
        # Plot joints
        ax.scatter(
            skeleton_points[:, 0],
            skeleton_points[:, 1],
            skeleton_points[:, 2],
            s=20,
            c=color,
            alpha=alpha,
            label=label,
        )

        # Plot connections (bones)
        for connection in self.skeleton_connections:
            if connection[0] < len(skeleton_points) and connection[1] < len(
                skeleton_points
            ):
                start_point = skeleton_points[connection[0]]
                end_point = skeleton_points[connection[1]]
                ax.plot(
                    [start_point[0], end_point[0]],
                    [start_point[1], end_point[1]],
                    [start_point[2], end_point[2]],
                    color=color,
                    alpha=alpha * 0.7,
                    linewidth=1,
                )

    def _set_axis_properties(self, ax, small=False):
        """Set consistent axis properties for 3D plots."""
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-2.0, 2.0)
        ax.set_zlim(-0.5, 2.0)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(azim=-75, elev=20)

        if small:
            # Reduce tick labels for smaller subplots
            ax.set_xticks([-2, 0, 2])
            ax.set_yticks([-2, 0, 2])
            ax.set_zticks([0, 1, 2])

    def verify_data_synchronization(
        self, trainer, max_subjects=6, frames_per_subject=5, create_videos=True
    ):
        """
        Create comprehensive visualization to verify MoCap and mmWave data synchronization.
        Shows train and test data for multiple subjects and frames, with optional video creation.
        """
        print("=" * 80)
        print("VERIFYING DATA SYNCHRONIZATION")
        print("=" * 80)

        verification_dir = os.path.join(self.output_dir, "verification")
        video_verification_dir = os.path.join(verification_dir, "videos")
        os.makedirs(verification_dir, exist_ok=True)
        if create_videos:
            os.makedirs(video_verification_dir, exist_ok=True)

        # Get data dimensions
        num_subjects = min(trainer.dataset.num_samples, max_subjects)
        train_length = trainer.dataset.train_length
        test_length = trainer.dataset.test_length

        print(
            f"Number of subjects: {trainer.dataset.num_samples} (showing {num_subjects})"
        )
        print(f"Train length: {train_length}")
        print(f"Test length: {test_length}")
        print(f"Total MoCap frames per subject: {trainer.dataset.total_length}")

        if create_videos:
            print("Video creation: ENABLED")
            print(f"Videos will be saved to: {video_verification_dir}")
        else:
            print("Video creation: DISABLED")

        # Verify each subject's data
        for subject_idx in range(num_subjects):
            print(f"\n--- Subject {subject_idx} ---")

            # Check train data
            self._verify_subject_data(
                trainer, subject_idx, "train", frames_per_subject, verification_dir
            )

            # Check test data
            self._verify_subject_data(
                trainer, subject_idx, "test", frames_per_subject, verification_dir
            )

            # Create videos for this subject
            if create_videos:
                self._create_subject_verification_videos(
                    trainer, subject_idx, video_verification_dir
                )

        # Create summary visualization
        self._create_verification_summary(trainer, verification_dir)

        print(f"\nVerification visualizations saved to: {verification_dir}")
        if create_videos:
            print(f"Verification videos saved to: {video_verification_dir}")
        print("=" * 80)

    def _verify_subject_data(
        self, trainer, subject_idx, split, frames_per_subject, output_dir
    ):
        """Verify data for a specific subject and split (train/test)."""

        if split == "train":
            # Get train data
            pc_data = trainer.dataset.pc_train[subject_idx]  # [T, P, 6]
            mocap_pquat = trainer.dataset.m_pquat_train[subject_idx]  # [T, J, 3, 3]
            mocap_trans = trainer.dataset.m_trans_train[subject_idx]  # [T, 3]
            data_length = trainer.dataset.train_length
        else:
            # Get test data
            pc_data = trainer.dataset.pc_test[subject_idx]  # [T, P, 6]
            mocap_pquat = trainer.dataset.m_pquat_test[subject_idx]  # [T, J, 3, 3]
            mocap_trans = trainer.dataset.m_trans_test[subject_idx]  # [T, 3]
            data_length = trainer.dataset.test_length

        print(
            f"  {split.upper()} - mmWave: {pc_data.shape}, MoCap pquat: {mocap_pquat.shape}, trans: {mocap_trans.shape}"
        )

        # Select frames to visualize (start, middle, end)
        if data_length <= frames_per_subject:
            frame_indices = list(range(data_length))
        else:
            frame_indices = [
                0,  # start
                data_length // 4,  # quarter
                data_length // 2,  # middle
                3 * data_length // 4,  # three quarters
                data_length - 1,  # end
            ][:frames_per_subject]

        # Convert MoCap data to skeleton for visualization
        betas = trainer.dataset.betas[subject_idx : subject_idx + 1]  # [1, 10]
        gender = trainer.dataset.gender[subject_idx : subject_idx + 1]  # [1]

        # Create tensors for selected frames
        selected_frames = len(frame_indices)
        pquat_tensor = torch.tensor(
            mocap_pquat[frame_indices][None, :],
            dtype=torch.float32,
            device=trainer.device,
        )  # [1, selected_frames, J, 3, 3]
        trans_tensor = torch.tensor(
            mocap_trans[frame_indices][None, :],
            dtype=torch.float32,
            device=trainer.device,
        )  # [1, selected_frames, 3]
        betas_tensor = torch.tensor(
            np.repeat(betas[:, None, :], selected_frames, axis=1),
            dtype=torch.float32,
            device=trainer.device,
        )  # [1, selected_frames, 10]
        gender_tensor = torch.tensor(
            np.repeat(gender[:, None, None], selected_frames, axis=1),
            dtype=torch.float32,
            device=trainer.device,
        )  # [1, selected_frames, 1]

        # Generate skeleton from MoCap data
        with torch.no_grad():
            _, gt_skeleton = trainer.cal_vs_from_qtbg(
                pquat_tensor,
                trans_tensor,
                betas_tensor,
                gender_tensor,
                1,
                selected_frames,
            )

        gt_skeleton = gt_skeleton.detach().cpu().numpy()[0]  # [selected_frames, 24, 3]

        # Create visualization for this subject and split
        fig = plt.figure(figsize=(20, 4 * frames_per_subject))
        fig.suptitle(
            f"Subject {subject_idx} - {split.upper()} Data Verification", fontsize=16
        )

        for i, frame_idx in enumerate(frame_indices):
            # mmWave point cloud
            ax1 = fig.add_subplot(frames_per_subject, 3, i * 3 + 1, projection="3d")
            pc_frame = pc_data[frame_idx]  # [P, 6]
            ax1.scatter(
                pc_frame[:, 0],
                pc_frame[:, 1],
                pc_frame[:, 2],
                s=10,
                c="blue",
                alpha=0.6,
            )
            ax1.set_title(f"mmWave Frame {frame_idx}\n({pc_frame.shape[0]} points)")
            self._set_axis_properties(ax1, small=True)

            # MoCap skeleton
            ax2 = fig.add_subplot(frames_per_subject, 3, i * 3 + 2, projection="3d")
            skeleton_frame = gt_skeleton[i]  # [24, 3]
            self._plot_skeleton(ax2, skeleton_frame, color="green", alpha=0.8)
            ax2.set_title(f"MoCap Frame {frame_idx}\nSkeleton")
            self._set_axis_properties(ax2, small=True)

            # Overlay comparison
            ax3 = fig.add_subplot(frames_per_subject, 3, i * 3 + 3, projection="3d")
            ax3.scatter(
                pc_frame[:, 0],
                pc_frame[:, 1],
                pc_frame[:, 2],
                s=5,
                c="blue",
                alpha=0.4,
                label="mmWave",
            )
            self._plot_skeleton(
                ax3, skeleton_frame, color="green", alpha=0.8, label="MoCap"
            )
            ax3.set_title(f"Overlay Frame {frame_idx}")
            ax3.legend()
            self._set_axis_properties(ax3, small=True)

        plt.tight_layout()

        filename = f"subject_{subject_idx:02d}_{split}_verification.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"    Saved: {filename}")

    def _create_verification_summary(self, trainer, output_dir):
        """Create a summary visualization showing data statistics."""

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Data Synchronization Summary", fontsize=16)

        # Plot 1: Frame counts comparison
        ax1 = axes[0, 0]
        subjects = range(trainer.dataset.num_samples)
        train_pc_counts = [trainer.dataset.pc_train[i].shape[0] for i in subjects]
        test_pc_counts = [trainer.dataset.pc_test[i].shape[0] for i in subjects]
        mocap_total = [trainer.dataset.total_length] * trainer.dataset.num_samples

        x = np.arange(len(subjects))
        width = 0.25

        ax1.bar(x - width, train_pc_counts, width, label="mmWave Train", alpha=0.7)
        ax1.bar(x, test_pc_counts, width, label="mmWave Test", alpha=0.7)
        ax1.bar(x + width, mocap_total, width, label="MoCap Total", alpha=0.7)

        ax1.set_xlabel("Subject Index")
        ax1.set_ylabel("Frame Count")
        ax1.set_title("Frame Counts by Subject")
        ax1.legend()
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"S{i}" for i in subjects])

        # Plot 2: Split ratios
        ax2 = axes[0, 1]
        total_mmw = [train_pc_counts[i] + test_pc_counts[i] for i in subjects]
        train_ratios = [
            train_pc_counts[i] / total_mmw[i] if total_mmw[i] > 0 else 0
            for i in subjects
        ]

        ax2.bar(subjects, train_ratios, alpha=0.7, color="skyblue")
        ax2.axhline(
            y=0.8, color="red", linestyle="--", alpha=0.7, label="Expected (80%)"
        )
        ax2.set_xlabel("Subject Index")
        ax2.set_ylabel("Train Ratio")
        ax2.set_title("Train/Test Split Ratios")
        ax2.legend()
        ax2.set_ylim(0, 1)

        # Plot 3: Point cloud sizes
        ax3 = axes[1, 0]
        pc_sizes_train = []
        pc_sizes_test = []

        for i in subjects:
            # Sample a few frames to get point cloud sizes
            train_sizes = [
                trainer.dataset.pc_train[i][j].shape[0]
                for j in range(0, min(10, trainer.dataset.pc_train[i].shape[0]), 2)
            ]
            test_sizes = [
                trainer.dataset.pc_test[i][j].shape[0]
                for j in range(0, min(10, trainer.dataset.pc_test[i].shape[0]), 2)
            ]
            pc_sizes_train.extend(train_sizes)
            pc_sizes_test.extend(test_sizes)

        ax3.hist(pc_sizes_train, bins=20, alpha=0.7, label="Train", density=True)
        ax3.hist(pc_sizes_test, bins=20, alpha=0.7, label="Test", density=True)
        ax3.axvline(
            x=trainer.dataset.pc_size,
            color="red",
            linestyle="--",
            label=f"Target ({trainer.dataset.pc_size})",
        )
        ax3.set_xlabel("Points per Frame")
        ax3.set_ylabel("Density")
        ax3.set_title("Point Cloud Size Distribution")
        ax3.legend()

        # Plot 4: Data info summary
        ax4 = axes[1, 1]
        ax4.axis("off")

        info_text = f"""
        Dataset Summary:

        Subjects: {trainer.dataset.num_samples}
        Total MoCap Length: {trainer.dataset.total_length}
        Train Length: {trainer.dataset.train_length}
        Test Length: {trainer.dataset.test_length}

        MoCap FPS: {trainer.dataset.mocap_fps}
        mmWave FPS: {trainer.dataset.mmwave_fps}

        Point Cloud Target Size: {trainer.dataset.pc_size}
        Joint Size: {trainer.dataset.joint_size}

        Split Method: {trainer.dataset.split_method}
        """

        ax4.text(
            0.1,
            0.9,
            info_text,
            transform=ax4.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontfamily="monospace",
        )

        plt.tight_layout()

        summary_path = os.path.join(output_dir, "synchronization_summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()

        print("Summary saved: synchronization_summary.png")

    def _create_subject_verification_videos(self, trainer, subject_idx, video_dir):
        """Create verification videos for a specific subject showing train and test data."""

        print(f"  Creating videos for Subject {subject_idx}...")

        # Create train video
        train_video_path = self._create_single_subject_video(
            trainer, subject_idx, "train", video_dir
        )

        # Create test video
        test_video_path = self._create_single_subject_video(
            trainer, subject_idx, "test", video_dir
        )

        # Create combined comparison video
        combined_video_path = self._create_combined_subject_video(
            trainer, subject_idx, video_dir
        )

        print(f"    Train video: {os.path.basename(train_video_path)}")
        print(f"    Test video: {os.path.basename(test_video_path)}")
        print(f"    Combined video: {os.path.basename(combined_video_path)}")

    def _create_single_subject_video(self, trainer, subject_idx, split, video_dir):
        """Create a video for a single subject and split (train/test)."""

        if split == "train":
            pc_data = trainer.dataset.pc_train[subject_idx]  # [T, P, 6]
            mocap_pquat = trainer.dataset.m_pquat_train[subject_idx]  # [T, J, 3, 3]
            mocap_trans = trainer.dataset.m_trans_train[subject_idx]  # [T, 3]
            data_length = trainer.dataset.train_length
        else:
            pc_data = trainer.dataset.pc_test[subject_idx]  # [T, P, 6]
            mocap_pquat = trainer.dataset.m_pquat_test[subject_idx]  # [T, J, 3, 3]
            mocap_trans = trainer.dataset.m_trans_test[subject_idx]  # [T, 3]
            data_length = trainer.dataset.test_length

        # Convert MoCap data to skeleton for all frames
        betas = trainer.dataset.betas[subject_idx : subject_idx + 1]  # [1, 10]
        gender = trainer.dataset.gender[subject_idx : subject_idx + 1]  # [1]

        # Create tensors for all frames
        pquat_tensor = torch.tensor(
            mocap_pquat[None, :], dtype=torch.float32, device=trainer.device
        )  # [1, T, J, 3, 3]
        trans_tensor = torch.tensor(
            mocap_trans[None, :], dtype=torch.float32, device=trainer.device
        )  # [1, T, 3]
        betas_tensor = torch.tensor(
            np.repeat(betas[:, None, :], data_length, axis=1),
            dtype=torch.float32,
            device=trainer.device,
        )  # [1, T, 10]
        gender_tensor = torch.tensor(
            np.repeat(gender[:, None, None], data_length, axis=1),
            dtype=torch.float32,
            device=trainer.device,
        )  # [1, T, 1]

        # Generate skeleton from MoCap data for all frames
        with torch.no_grad():
            _, gt_skeleton = trainer.cal_vs_from_qtbg(
                pquat_tensor, trans_tensor, betas_tensor, gender_tensor, 1, data_length
            )

        gt_skeleton = gt_skeleton.detach().cpu().numpy()[0]  # [T, 24, 3]

        # Create video frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_files = []

            # Sample frames for video (to avoid too long videos)
            max_frames = min(data_length, 300)  # Limit to ~30 seconds at 10fps
            frame_indices = list(range(0, data_length))[:max_frames]

            for i, frame_idx in enumerate(frame_indices):
                frame_path = self._create_verification_frame(
                    pc_data[frame_idx],
                    gt_skeleton[frame_idx],
                    subject_idx,
                    split,
                    frame_idx,
                    i,
                    temp_dir,
                )
                frame_files.append(frame_path)

            # Create video
            video_filename = f"subject_{subject_idx:02d}_{split}_verification.mp4"
            video_path = os.path.join(video_dir, video_filename)
            self._create_video_from_frames(frame_files, video_path, fps=10)

        return video_path

    def _create_combined_subject_video(self, trainer, subject_idx, video_dir):
        """Create a side-by-side comparison video showing train and test data."""

        # Get data for both splits
        pc_train = trainer.dataset.pc_train[subject_idx]
        pc_test = trainer.dataset.pc_test[subject_idx]

        mocap_pquat_train = trainer.dataset.m_pquat_train[subject_idx]
        mocap_pquat_test = trainer.dataset.m_pquat_test[subject_idx]

        mocap_trans_train = trainer.dataset.m_trans_train[subject_idx]
        mocap_trans_test = trainer.dataset.m_trans_test[subject_idx]

        # Generate skeletons for both splits
        betas = trainer.dataset.betas[subject_idx : subject_idx + 1]
        gender = trainer.dataset.gender[subject_idx : subject_idx + 1]

        # Train skeletons
        pquat_tensor_train = torch.tensor(
            mocap_pquat_train[None, :], dtype=torch.float32, device=trainer.device
        )
        trans_tensor_train = torch.tensor(
            mocap_trans_train[None, :], dtype=torch.float32, device=trainer.device
        )
        betas_tensor_train = torch.tensor(
            np.repeat(betas[:, None, :], trainer.dataset.train_length, axis=1),
            dtype=torch.float32,
            device=trainer.device,
        )
        gender_tensor_train = torch.tensor(
            np.repeat(gender[:, None, None], trainer.dataset.train_length, axis=1),
            dtype=torch.float32,
            device=trainer.device,
        )

        # Test skeletons
        pquat_tensor_test = torch.tensor(
            mocap_pquat_test[None, :], dtype=torch.float32, device=trainer.device
        )
        trans_tensor_test = torch.tensor(
            mocap_trans_test[None, :], dtype=torch.float32, device=trainer.device
        )
        betas_tensor_test = torch.tensor(
            np.repeat(betas[:, None, :], trainer.dataset.test_length, axis=1),
            dtype=torch.float32,
            device=trainer.device,
        )
        gender_tensor_test = torch.tensor(
            np.repeat(gender[:, None, None], trainer.dataset.test_length, axis=1),
            dtype=torch.float32,
            device=trainer.device,
        )

        with torch.no_grad():
            _, gt_skeleton_train = trainer.cal_vs_from_qtbg(
                pquat_tensor_train,
                trans_tensor_train,
                betas_tensor_train,
                gender_tensor_train,
                1,
                trainer.dataset.train_length,
            )
            _, gt_skeleton_test = trainer.cal_vs_from_qtbg(
                pquat_tensor_test,
                trans_tensor_test,
                betas_tensor_test,
                gender_tensor_test,
                1,
                trainer.dataset.test_length,
            )

        gt_skeleton_train = gt_skeleton_train.detach().cpu().numpy()[0]
        gt_skeleton_test = gt_skeleton_test.detach().cpu().numpy()[0]

        # Create combined video frames
        with tempfile.TemporaryDirectory() as temp_dir:
            frame_files = []

            # Use the shorter sequence length for synchronization
            min_length = min(trainer.dataset.train_length, trainer.dataset.test_length)
            max_frames = min(min_length, 300)  # Limit video length
            frame_indices = list(range(0, min_length))[:max_frames]

            for i, frame_idx in enumerate(frame_indices):
                frame_path = self._create_combined_verification_frame(
                    pc_train[frame_idx],
                    gt_skeleton_train[frame_idx],
                    pc_test[frame_idx],
                    gt_skeleton_test[frame_idx],
                    subject_idx,
                    frame_idx,
                    i,
                    temp_dir,
                )
                frame_files.append(frame_path)

            # Create video
            video_filename = f"subject_{subject_idx:02d}_combined_verification.mp4"
            video_path = os.path.join(video_dir, video_filename)
            self._create_video_from_frames(frame_files, video_path, fps=10)

        return video_path

    def _create_verification_frame(
        self,
        pc_frame,
        skeleton_frame,
        subject_idx,
        split,
        original_frame_idx,
        video_frame_idx,
        output_dir,
    ):
        """Create a single verification frame showing mmWave and MoCap data."""

        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(
            f"Subject {subject_idx} - {split.upper()} - Frame {original_frame_idx}",
            fontsize=14,
        )

        # mmWave point cloud
        ax1 = fig.add_subplot(131, projection="3d")
        ax1.scatter(
            pc_frame[:, 0], pc_frame[:, 1], pc_frame[:, 2], s=10, c="blue", alpha=0.6
        )
        ax1.set_title(f"mmWave\n({pc_frame.shape[0]} points)")
        self._set_axis_properties(ax1)

        # MoCap skeleton
        ax2 = fig.add_subplot(132, projection="3d")
        self._plot_skeleton(ax2, skeleton_frame, color="green", alpha=0.8)
        ax2.set_title("MoCap Skeleton")
        self._set_axis_properties(ax2)

        # Overlay comparison
        ax3 = fig.add_subplot(133, projection="3d")
        ax3.scatter(
            pc_frame[:, 0],
            pc_frame[:, 1],
            pc_frame[:, 2],
            s=5,
            c="blue",
            alpha=0.4,
            label="mmWave",
        )
        self._plot_skeleton(
            ax3, skeleton_frame, color="green", alpha=0.8, label="MoCap"
        )
        ax3.set_title("Overlay")
        ax3.legend()
        self._set_axis_properties(ax3)

        plt.tight_layout()

        filename = f"frame_{video_frame_idx:04d}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        return filepath

    def _create_combined_verification_frame(
        self,
        pc_train,
        skeleton_train,
        pc_test,
        skeleton_test,
        subject_idx,
        original_frame_idx,
        video_frame_idx,
        output_dir,
    ):
        """Create a combined frame showing train and test data side by side."""

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle(
            f"Subject {subject_idx} - Train vs Test Comparison - Frame {original_frame_idx}",
            fontsize=16,
        )

        # Train data - top row
        ax1 = fig.add_subplot(2, 3, 1, projection="3d")
        ax1.scatter(
            pc_train[:, 0], pc_train[:, 1], pc_train[:, 2], s=8, c="blue", alpha=0.6
        )
        ax1.set_title(f"TRAIN - mmWave\n({pc_train.shape[0]} points)")
        self._set_axis_properties(ax1, small=True)

        ax2 = fig.add_subplot(2, 3, 2, projection="3d")
        self._plot_skeleton(ax2, skeleton_train, color="green", alpha=0.8)
        ax2.set_title("TRAIN - MoCap")
        self._set_axis_properties(ax2, small=True)

        ax3 = fig.add_subplot(2, 3, 3, projection="3d")
        ax3.scatter(
            pc_train[:, 0],
            pc_train[:, 1],
            pc_train[:, 2],
            s=4,
            c="blue",
            alpha=0.4,
            label="mmWave",
        )
        self._plot_skeleton(
            ax3, skeleton_train, color="green", alpha=0.8, label="MoCap"
        )
        ax3.set_title("TRAIN - Overlay")
        ax3.legend()
        self._set_axis_properties(ax3, small=True)

        # Test data - bottom row
        ax4 = fig.add_subplot(2, 3, 4, projection="3d")
        ax4.scatter(
            pc_test[:, 0], pc_test[:, 1], pc_test[:, 2], s=8, c="orange", alpha=0.6
        )
        ax4.set_title(f"TEST - mmWave\n({pc_test.shape[0]} points)")
        self._set_axis_properties(ax4, small=True)

        ax5 = fig.add_subplot(2, 3, 5, projection="3d")
        self._plot_skeleton(ax5, skeleton_test, color="red", alpha=0.8)
        ax5.set_title("TEST - MoCap")
        self._set_axis_properties(ax5, small=True)

        ax6 = fig.add_subplot(2, 3, 6, projection="3d")
        ax6.scatter(
            pc_test[:, 0],
            pc_test[:, 1],
            pc_test[:, 2],
            s=4,
            c="orange",
            alpha=0.4,
            label="mmWave",
        )
        self._plot_skeleton(ax6, skeleton_test, color="red", alpha=0.8, label="MoCap")
        ax6.set_title("TEST - Overlay")
        ax6.legend()
        self._set_axis_properties(ax6, small=True)

        plt.tight_layout()

        filename = f"combined_frame_{video_frame_idx:04d}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=100, bbox_inches="tight")
        plt.close()

        return filepath


# Enhanced usage functions
def create_training_visualization(trainer, step, output_dir="./visualization"):
    """
    Create training visualizations including videos.
    """
    visualizer = TrainingVisualizer(output_dir)

    # Create training visualizations every 500 steps
    train_results = visualizer.visualize_batch_sample(
        trainer, step, save_every_n_steps=500
    )

    # Create evaluation visualizations every 1000 steps
    eval_results = visualizer.visualize_evaluation_data(
        trainer, step, save_every_n_steps=1000
    )

    return train_results, eval_results
