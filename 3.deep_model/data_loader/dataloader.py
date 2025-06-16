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
        # Collect mocap dicts
        data_list = []
        for p in sorted(paths):
            with open(p, "rb") as f:
                data_list.append(pickle.load(f))

        # Downsample step: e.g., 120fps -> 10fps gives step=12
        step = self.mocap_fps // self.mmwave_fps
        # Number of output frames at mmWave rate
        raw_frames = data_list[0]["trans"].shape[0]
        total_frames = raw_frames // step
        # Number of rotation joints (fullpose length is total_frames*3)
        J = data_list[0]["fullpose"].shape[1] // 3

        N = len(data_list)
        self.betas = np.stack([d["betas"][:10] for d in data_list], axis=0)
        # If mocap dict contains gender use it, else default to zeros
        self.gender = np.array(
            [d.get("gender", 0) for d in data_list], dtype=np.float32
        )

        smpl_map = list(range(15)) + [16, 15, 17, 18, 19, 20, 21]
        num_smpl = len(smpl_map)
        self.joint_size = num_smpl

        # Allocate buffers
        self.trans = np.zeros((N, total_frames, 3), dtype=np.float32)
        self.pquat = np.zeros((N, total_frames, num_smpl, 3, 3), dtype=np.float32)

        # Fill buffers
        for i, d in enumerate(data_list):
            # Sample every `step` frames
            trans_ds = d["trans"][::step][:total_frames]
            pose_ds = d["fullpose"][::step][:total_frames]

            self.trans[i] = trans_ds
            # Convert Rodrigues vectors to rotation matrices
            rot_mats = Rotation.from_rotvec(pose_ds.reshape(-1, 3)).as_matrix()
            rot_mats = rot_mats.reshape(total_frames, J, 3, 3)
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
        split = int(self.total_length * (1 - self.test_split))

        # mocap
        self.m_trans_train = self.trans[:, :split]
        self.m_trans_test = self.trans[:, split:]
        self.m_pquat_train = self.pquat[:, :split]
        self.m_pquat_test = self.pquat[:, split:]

        # mmWave
        self.pc_train = [seq[:split] for seq in self.pc_train]
        self.pc_test = [seq[split:] for seq in self.pc_test]

        self.train_length = split
        self.test_length = self.total_length - split

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
