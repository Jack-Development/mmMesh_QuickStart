import time
import pickle
import numpy as np
import multiprocessing as mp
from scipy.spatial.transform import Rotation
from typing import List, Optional


class DataLoader:
    def __init__(
        self,
        mocap_paths: List[str],
        mmwave_train_path: str,
        mmwave_test_path: str,
        batch_size: int,
        seq_length: int,
        pc_size: int,
        test_split_ratio: float = 0.2,
        split_method: str = "sequential",
        prefetch_size: int = 128,
        test_buffer: int = 2,
        num_workers: int = 4,
        device: str = "cpu",
        smpl_model: Optional[object] = None,
        mocap_fps: int = 120,
        mmwave_fps: int = 10,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.pc_size = pc_size
        self.test_split = test_split_ratio
        self.split_method = split_method
        self.prefetch_size = prefetch_size
        self.test_buffer = test_buffer
        self.num_workers = num_workers
        self.device = device
        self.smpl_model = smpl_model
        self.mocap_fps = mocap_fps
        self.mmwave_fps = mmwave_fps

        # 1) Load raw data
        self._load_mocap(mocap_paths)
        self._load_mmwave(mmwave_train_path, mmwave_test_path)

        # 2) Split time axis
        self._split_data()

        # 3) Prefetch buffers
        self._init_queues()
        self._start_workers()

    def _load_mocap(self, paths: List[str]):
        """Load mocap .pkl files into arrays: trans, pquat, betas, gender."""
        data_list = []
        for p in sorted(paths):
            with open(p, "rb") as f:
                data_list.append(pickle.load(f))

        step = self.mocap_fps // self.mmwave_fps

        # figure out how many down-sampled frames each file has,
        # then pick the smallest so every file can fit
        down_lengths = [d["trans"].shape[0] // step for d in data_list]
        total_frames = min(down_lengths)

        J = data_list[0]["fullpose"].shape[1] // 3

        N = len(data_list)
        self.betas = np.stack([d["betas"][:10] for d in data_list], axis=0)
        self.gender = np.array(
            [d.get("gender", 0) for d in data_list], dtype=np.float32
        )

        smpl_map = list(range(15)) + [16, 15, 17, 18, 19, 20, 21]
        num_smpl = len(smpl_map)
        self.joint_size = num_smpl

        # now allocate two uniform buffers
        self.trans = np.zeros((N, total_frames, 3), dtype=np.float32)
        self.pquat = np.zeros((N, total_frames, num_smpl, 3, 3), dtype=np.float32)

        for i, d in enumerate(data_list):
            trans_ds = d["trans"][::step][:total_frames]  # shape: (total_frames, 3)
            pose_ds = d["fullpose"][::step][:total_frames]  # shape: (total_frames, J*3)

            self.trans[i] = trans_ds

            rot_mats = (
                Rotation.from_rotvec(pose_ds.reshape(-1, 3))
                .as_matrix()
                .reshape(total_frames, J, 3, 3)
            )
            self.pquat[i] = rot_mats[:, smpl_map, :, :]

        self.num_samples = N
        self.total_length = total_frames

    def _load_mmwave(self, train_path: str, test_path: str):
        """Load mmWave point clouds from pickled .bin files."""
        with open(train_path, "rb") as f:
            self.pc_train = pickle.load(f)
        with open(test_path, "rb") as f:
            self.pc_test = pickle.load(f)
        # Expect shapes [N][T][num_points][6]

    def _split_data(self):
        """Split mocap and mmWave along temporal axis into train/test."""
        if self.split_method == "sequential":
            self._split_sequential()
        elif self.split_method == "end":
            self._split_end()
        else:
            raise ValueError(f"Unknown split method: {self.split_method}")

        # Debug shape of Mocap and mmWave data
        print("\n=== DEBUG: DataLoader split ===")
        print(f"Total subjects: {self.num_samples}")
        print(f"Test split ratio: {self.test_split}")
        print(f"Total mocap length: {self.total_length}")
        print(f"Train length: {self.train_length}, Test length: {self.test_length}")
        print(f"Train mocap shape: {self.m_trans_train.shape}")
        print(f"Test mocap shape: {self.m_trans_test.shape}")
        print(f"Train mmWave shape: {[seq.shape for seq in self.pc_train]}")
        print(f"Test mmWave shape: {[seq.shape for seq in self.pc_test]}")
        print("=== END DEBUG ===\n")

    def _split_sequential(self):
        """Sequential split: overall temporal split (current method)."""
        # Mocap-based split
        mocap_split = int(self.total_length * (1 - self.test_split))
        mocap_test = self.total_length - mocap_split

        # How many frames mmWave actually has in each list
        mmw_train_lens = [seq.shape[0] for seq in self.pc_train]
        mmw_test_lens = [seq.shape[0] for seq in self.pc_test]

        # Final lengths = min(mocap, mmWave) for each phase
        self.train_length = min(mocap_split, min(mmw_train_lens))
        self.test_length = min(mocap_test, min(mmw_test_lens))

        # Crop both modalities exactly to those lengths
        # Mocap
        self.m_trans_train = self.trans[:, : self.train_length]
        self.m_trans_test = self.trans[:, -self.test_length :]
        self.m_pquat_train = self.pquat[:, : self.train_length]
        self.m_pquat_test = self.pquat[:, -self.test_length :]

        # mmWave
        # self.pc_train = [seq[: self.train_length] for seq in self.pc_train]
        # self.pc_test = [seq[: self.test_length] for seq in self.pc_test]

    def _split_end(self):
        """End split: per-clip split taking last test_split percentage from each clip."""
        print("\n=== DEBUG: _split_end method ===")
        print(f"Total subjects: {self.num_samples}")
        print(f"Test split ratio: {self.test_split}")
        print(f"Total mocap length: {self.total_length}")

        # For each subject, split their individual timeline
        train_lengths = []
        test_lengths = []

        for s in range(self.num_samples):
            # Get available lengths for this subject
            mocap_len = self.total_length
            mmw_train_len = self.pc_train[s].shape[0]
            mmw_test_len = self.pc_test[s].shape[0]

            # Use minimum available length as the clip's total length
            clip_length = min(mocap_len, mmw_train_len + mmw_test_len)

            # Split this clip's timeline
            test_count = int(self.test_split * clip_length)
            test_count = max(test_count, 1) if clip_length > 1 else 0
            train_count = clip_length - test_count

            train_lengths.append(train_count)
            test_lengths.append(test_count)

            print(
                f"Subject {s:2d}: mocap={mocap_len}, mmw_train={mmw_train_len}, mmw_test={mmw_test_len}"
            )
            print(
                f"           clip_length={clip_length}, train={train_count}, test={test_count}"
            )

        # Use minimum lengths across all subjects for consistent batching
        self.train_length = min(train_lengths)
        self.test_length = min(test_lengths)

        print(f"\nFinal lengths - train: {self.train_length}, test: {self.test_length}")

        # Split each subject's data individually
        m_trans_train_list = []
        m_trans_test_list = []
        m_pquat_train_list = []
        m_pquat_test_list = []

        for s in range(self.num_samples):
            # Get this subject's available length
            clip_length = min(
                self.total_length, self.pc_train[s].shape[0] + self.pc_test[s].shape[0]
            )

            # Split mocap data for this subject
            test_start = clip_length - self.test_length

            m_trans_train_list.append(self.trans[s, : self.train_length])
            m_trans_test_list.append(
                self.trans[s, test_start : test_start + self.test_length]
            )
            m_pquat_train_list.append(self.pquat[s, : self.train_length])
            m_pquat_test_list.append(
                self.pquat[s, test_start : test_start + self.test_length]
            )

            print(
                f"Subject {s:2d} mocap split: train frames 0-{self.train_length-1}, test frames {test_start}-{test_start + self.test_length - 1}"
            )

        # Stack back into arrays
        self.m_trans_train = np.stack(m_trans_train_list, axis=0)
        self.m_trans_test = np.stack(m_trans_test_list, axis=0)
        self.m_pquat_train = np.stack(m_pquat_train_list, axis=0)
        self.m_pquat_test = np.stack(m_pquat_test_list, axis=0)

        print("\nFinal arrays:")
        print(f"m_trans_train: {self.m_trans_train.shape}")
        print(f"m_trans_test: {self.m_trans_test.shape}")
        print(f"m_pquat_train: {self.m_pquat_train.shape}")
        print(f"m_pquat_test: {self.m_pquat_test.shape}")
        print("=== END DEBUG ===\n")

    def _init_queues(self):
        self.flag = mp.Value("b", True)
        self.q_train = mp.Queue(maxsize=self.prefetch_size)
        self.q_test = mp.Queue(maxsize=self.test_buffer)

    def _start_workers(self):
        self.workers = []
        # train workers
        for _ in range(self.num_workers):
            p = mp.Process(target=self._worker_train)
            p.daemon = True
            p.start()
            self.workers.append(p)
        # test worker
        p = mp.Process(target=self._worker_test)
        p.daemon = True
        p.start()
        self.workers.append(p)

    def _worker_train(self):
        rng = np.random.default_rng()
        while self.flag.value:
            try:
                self.q_train.put(self._make_batch(rng), timeout=0.5)
            except mp.queues.Full:
                time.sleep(0.1)

    def _worker_test(self):
        while self.flag.value:
            try:
                self.q_test.put(self._make_test(), timeout=1.0)
            except mp.queues.Full:
                time.sleep(0.5)

    def _pad_or_sample(self, pts: np.ndarray, rng: np.random.Generator):
        n = pts.shape[0]
        if n >= self.pc_size:
            idx = rng.choice(n, self.pc_size, replace=False)
            return pts[idx]
        # pad + duplicate
        out = np.zeros((self.pc_size, pts.shape[1]), dtype=np.float32)
        idx = rng.choice(self.pc_size, n, replace=False)
        out[idx] = pts
        rem = [i for i in range(self.pc_size) if i not in idx]
        dup = rng.choice(n, len(rem), replace=True)
        out[rem] = pts[dup]
        return out

    def _make_batch(self, rng):
        seq = self.seq_length
        subjects = rng.integers(0, self.num_samples, self.batch_size)
        starts = rng.integers(0, self.train_length - seq + 1, self.batch_size)

        pc_batch = []
        pquat_batch = []
        trans_batch = []
        betas_batch = []
        gender_batch = []

        for s, st in zip(subjects, starts):
            # mocap
            trans_batch.append(self.m_trans_train[s, st : st + seq])
            pquat_batch.append(self.m_pquat_train[s, st : st + seq])
            betas_batch.append(np.repeat(self.betas[s][None], seq, axis=0))
            gender_batch.append(np.repeat([self.gender[s][None]], seq, axis=0))
            # mmWave
            pc_seq = []
            for t in range(st, st + seq):
                pc_seq.append(self._pad_or_sample(self.pc_train[s][t], rng))
            pc_batch.append(pc_seq)

        return (
            np.stack(pc_batch).astype(np.float32),
            np.stack(pquat_batch).astype(np.float32),
            np.stack(trans_batch).astype(np.float32),
            np.stack(betas_batch).astype(np.float32),
            np.stack(gender_batch).astype(np.float32),
        )

    def _make_test(self):
        all_pc = []
        for s in range(self.num_samples):
            seq_pc = []
            for t in range(self.test_length):
                seq_pc.append(
                    self._pad_or_sample(self.pc_test[s][t], np.random.default_rng())
                )
            all_pc.append(seq_pc)
        return np.stack(all_pc).astype(np.float32)

    def next_batch(self, timeout: float = None):
        return self.q_train.get(timeout=timeout)

    def get_test(self, timeout: float = None):
        return self.q_test.get(timeout=timeout)

    def close(self):
        """Gracefully shutdown workers and clear queues."""
        self.flag.value = False
        time.sleep(1)
        for q in (self.q_train, self.q_test):
            while not q.empty():
                q.get_nowait()
        for p in self.workers:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
