from pathlib import Path
import numpy as np
import os
import glob
from tqdm import tqdm
import configuration as cfg
from concurrent.futures import ProcessPoolExecutor, as_completed
from visualize import create_dataset_visualizations


class FrameConfig:
    def __init__(self):
        # configs in configuration.py
        self.numTxAntennas = cfg.NUM_TX
        self.numRxAntennas = cfg.NUM_RX
        self.numLoopsPerFrame = cfg.LOOPS_PER_FRAME
        self.numADCSamples = cfg.ADC_SAMPLES
        self.numAngleBins = cfg.NUM_ANGLE_BINS

        self.numChirpsPerFrame = self.numTxAntennas * self.numLoopsPerFrame
        self.numRangeBins = self.numADCSamples
        self.numDopplerBins = self.numLoopsPerFrame

        # calculate size of one chirp in short
        self.chirpSize = self.numRxAntennas * self.numADCSamples
        # calculate size of one chirp loop in short. 3Tx has three chirps in one loop for TDM
        self.chirpLoopSize = self.chirpSize * self.numTxAntennas
        # calculate size of one frame in short
        self.frameSize = self.chirpLoopSize * self.numLoopsPerFrame


class PointCloudProcessCFG:
    def __init__(self):
        self.frameConfig = FrameConfig()
        self.enableStaticClutterRemoval = True
        self.EnergyTop128 = False
        self.EnergyTop256 = True
        self.RangeCut = True
        self.outputVelocity = True
        self.outputSNR = True
        self.outputRange = True
        self.outputInMeter = True

        # 0,1,2 for x,y,z
        dim = 3
        if self.outputVelocity:
            self.velocityDim = dim
            dim += 1
        if self.outputSNR:
            self.SNRDim = dim
            dim += 1
        if self.outputRange:
            self.rangeDim = dim
            dim += 1
        self.couplingSignatureBinFrontIdx = 5
        self.couplingSignatureBinRearIdx = 4
        self.sumCouplingSignatureArray = np.zeros(
            (
                self.frameConfig.numTxAntennas,
                self.frameConfig.numRxAntennas,
                self.couplingSignatureBinFrontIdx + self.couplingSignatureBinRearIdx,
            ),
            dtype=complex,
        )


class RawDataReader:
    def __init__(self, path):
        self.path = path
        self.ADCBinFile = open(path, "rb")

    def getNextFrame(self, frameconfig):
        bytes_to_read = frameconfig.frameSize * 4
        buf = self.ADCBinFile.read(bytes_to_read)
        if len(buf) < bytes_to_read:
            return None
        return np.frombuffer(buf, dtype=np.int16)

    def close(self):
        self.ADCBinFile.close()


def bin2np_frame(bin_frame):
    np_frame = np.zeros(shape=(len(bin_frame) // 2), dtype=np.complex_)
    np_frame[0::2] = bin_frame[0::4] + 1j * bin_frame[2::4]
    np_frame[1::2] = bin_frame[1::4] + 1j * bin_frame[3::4]
    return np_frame


def frameReshape(frame, frameConfig):
    frameWithChirp = np.reshape(
        frame,
        (
            frameConfig.numLoopsPerFrame,
            frameConfig.numTxAntennas,
            frameConfig.numRxAntennas,
            -1,
        ),
    )
    return frameWithChirp.transpose(1, 2, 0, 3)


def rangeFFT(reshapedFrame, frameConfig):
    windowedBins1D = reshapedFrame * np.hamming(frameConfig.numADCSamples)
    rangeFFTResult = np.fft.fft(windowedBins1D)
    return rangeFFTResult


def clutter_removal(input_val, axis=0):
    # Reorder the axes
    reordering = np.arange(len(input_val.shape))
    reordering[0] = axis
    reordering[axis] = 0
    input_val = input_val.transpose(reordering)
    # Apply static clutter removal
    mean = input_val.mean(0)
    output_val = input_val - mean
    return output_val.transpose(reordering)


def dopplerFFT(rangeResult, frameConfig):
    windowedBins2D = rangeResult * np.reshape(
        np.hamming(frameConfig.numLoopsPerFrame), (1, 1, -1, 1)
    )
    dopplerFFTResult = np.fft.fft(windowedBins2D, axis=2)
    dopplerFFTResult = np.fft.fftshift(dopplerFFTResult, axes=2)
    return dopplerFFTResult


def naive_xyz(virtual_ant, num_tx=3, num_rx=4, fft_size=64):
    assert num_tx > 2, "need a config for more than 2 TXs"
    num_detected_obj = virtual_ant.shape[1]
    azimuth_ant = virtual_ant[: 2 * num_rx, :]
    azimuth_ant_padded = np.zeros(shape=(fft_size, num_detected_obj), dtype=np.complex_)
    azimuth_ant_padded[: 2 * num_rx, :] = azimuth_ant

    # Process azimuth information
    azimuth_fft = np.fft.fft(azimuth_ant_padded, axis=0)
    k_max = np.argmax(np.abs(azimuth_fft), axis=0)
    peak_1 = np.zeros_like(k_max, dtype=np.complex_)
    for i in range(len(k_max)):
        peak_1[i] = azimuth_fft[k_max[i], i]

    k_max[k_max > (fft_size // 2) - 1] = k_max[k_max > (fft_size // 2) - 1] - fft_size
    wx = 2 * np.pi / fft_size * k_max
    x_vector = wx / np.pi

    # Zero pad elevation
    elevation_ant = virtual_ant[2 * num_rx :, :]
    elevation_ant_padded = np.zeros(
        shape=(fft_size, num_detected_obj), dtype=np.complex_
    )
    elevation_ant_padded[:num_rx, :] = elevation_ant

    # Process elevation information
    elevation_fft = np.fft.fft(elevation_ant_padded, axis=0)
    elevation_max = np.argmax(
        np.log2(np.abs(elevation_fft)), axis=0
    )  # shape = (num_detected_obj, )
    peak_2 = np.zeros_like(elevation_max, dtype=np.complex_)
    for i in range(len(elevation_max)):
        peak_2[i] = elevation_fft[elevation_max[i], i]

    # Calculate elevation phase shift
    wz = np.angle(peak_1 * peak_2.conj() * np.exp(1j * 2 * wx))
    z_vector = wz / np.pi
    ypossible = 1 - x_vector**2 - z_vector**2
    y_vector = ypossible
    x_vector[ypossible < 0] = 0
    z_vector[ypossible < 0] = 0
    y_vector[ypossible < 0] = 0
    y_vector = np.sqrt(y_vector)
    return x_vector, y_vector, z_vector


def frame2pointcloud(frame, pointCloudProcessCFG):
    frameConfig = pointCloudProcessCFG.frameConfig
    reshapedFrame = frameReshape(frame, frameConfig)
    rangeResult = rangeFFT(reshapedFrame, frameConfig)
    if pointCloudProcessCFG.enableStaticClutterRemoval:
        rangeResult = clutter_removal(rangeResult, axis=2)
    dopplerResult = dopplerFFT(rangeResult, frameConfig)

    dopplerResultSumAllAntenna = np.sum(dopplerResult, axis=(0, 1))
    dopplerResultInDB = np.log10(np.absolute(dopplerResultSumAllAntenna))

    if pointCloudProcessCFG.RangeCut:
        dopplerResultInDB[:, :25] = -100
        dopplerResultInDB[:, 125:] = -100

    cfarResult = np.zeros(dopplerResultInDB.shape, bool)
    if pointCloudProcessCFG.EnergyTop128:
        top_size = 128
        energyThre128 = np.partition(
            dopplerResultInDB.ravel(), 128 * 256 - top_size - 1
        )[128 * 256 - top_size - 1]
        cfarResult[dopplerResultInDB > energyThre128] = True
    elif pointCloudProcessCFG.EnergyTop256:
        top_size = 256
        energyThre128 = np.partition(
            dopplerResultInDB.ravel(), 128 * 256 - top_size - 1
        )[128 * 256 - top_size - 1]
        cfarResult[dopplerResultInDB > energyThre128] = True

    det_peaks_indices = np.argwhere(cfarResult)
    R = det_peaks_indices[:, 1].astype(np.float64)
    V = (det_peaks_indices[:, 0] - frameConfig.numDopplerBins // 2).astype(np.float64)
    if pointCloudProcessCFG.outputInMeter:
        R *= cfg.RANGE_RESOLUTION
        V *= cfg.DOPPLER_RESOLUTION
    energy = dopplerResultInDB[cfarResult]

    AOAInput = dopplerResult[:, :, cfarResult]
    AOAInput = AOAInput.reshape(12, -1)

    if AOAInput.shape[1] == 0:
        return np.array([]).reshape(6, 0)
    x_vec, y_vec, z_vec = naive_xyz(AOAInput)

    x, y, z = x_vec * R, y_vec * R, z_vec * R
    pointCloud = np.concatenate((x, y, z, V, energy, R))
    pointCloud = np.reshape(pointCloud, (6, -1))
    pointCloud = pointCloud[:, y_vec != 0]
    return pointCloud


def apply_coordinate_flips(points):
    if cfg.FLIP_X:
        points[:, 0] = -points[:, 0]  # Flip X coordinate

    if cfg.FLIP_Y:
        points[:, 1] = -points[:, 1]  # Flip Y coordinate

    return points


def reg_data(data, pc_size):
    pc_tmp = np.zeros((pc_size, 6), dtype=np.float32)
    pc_no = data.shape[0]
    if pc_no < pc_size:
        fill_list = np.random.choice(pc_size, size=pc_no, replace=False)
        fill_set = set(fill_list)
        pc_tmp[fill_list] = data
        dupl_list = [x for x in range(pc_size) if x not in fill_set]
        dupl_pc = np.random.choice(pc_no, size=len(dupl_list), replace=True)
        pc_tmp[dupl_list] = data[dupl_pc]
    else:
        pc_list = np.random.choice(pc_no, size=pc_size, replace=False)
        pc_tmp = data[pc_list]
    return pc_tmp


def process_bin_file(
    bin_path: Path, pc_size: int, shift_arr: np.ndarray, cfg_obj: PointCloudProcessCFG
) -> np.ndarray:
    """
    Read frames until EOF from a .bin file, convert each to a fixed-size
    (pc_size x 6) array, and return a (n_frames, pc_size, 6) ndarray.
    """
    reader = RawDataReader(str(bin_path))
    frames = []

    while True:
        try:
            raw_frame = reader.getNextFrame(cfg_obj.frameConfig)
        except (StopIteration, EOFError):
            break
        if raw_frame is None:
            break

        np_frame = bin2np_frame(raw_frame)
        pc = frame2pointcloud(np_frame, cfg_obj)

        if pc.size == 0:
            sampled = np.zeros((pc_size, 6), dtype=np.float32)
        else:
            raw_pts = pc.T
            raw_pts[:, :3] += shift_arr
            raw_pts = apply_coordinate_flips(raw_pts)
            sampled = reg_data(raw_pts, pc_size)

        frames.append(sampled)

    reader.close()

    if not frames:
        raise ValueError(f"No frames extracted from {bin_path}")
    return np.stack(frames, axis=0)


def split_sequential(dataset, train_count, min_frames):
    """
    Sequential split: Fill train set with complete files sequentially,
    then put remaining files in test set.
    """
    train_splits = []
    test_splits = []
    current_train_frames = 0

    for i, arr in enumerate(dataset):
        arr_trunc = arr[:min_frames]  # Truncate to min_frames

        # If adding this entire file to train won't exceed train_count
        if current_train_frames + min_frames <= train_count:
            train_splits.append(arr_trunc)
            current_train_frames += min_frames
        else:
            # If we still need some frames for train set
            if current_train_frames < train_count:
                remaining_train_needed = train_count - current_train_frames
                train_splits.append(arr_trunc[:remaining_train_needed])
                test_splits.append(arr_trunc[remaining_train_needed:])
                current_train_frames = train_count
            else:
                # All remaining files go to test
                test_splits.append(arr_trunc)

    # Pad arrays to have consistent shapes for stacking
    if train_splits:
        # All train files should have the same length (min_frames or partial)
        max_train_len = max(arr.shape[0] for arr in train_splits)
        train_padded = []
        for arr in train_splits:
            if arr.shape[0] < max_train_len:
                # This should only happen for the last partial file
                padded = np.zeros(
                    (max_train_len, arr.shape[1], arr.shape[2]), dtype=arr.dtype
                )
                padded[: arr.shape[0]] = arr
                train_padded.append(padded)
            else:
                train_padded.append(arr)
        train_array = np.concatenate(train_padded, axis=0).reshape(
            -1, arr.shape[1], arr.shape[2]
        )
    else:
        train_array = np.empty(
            (0, dataset[0].shape[1], dataset[0].shape[2]), dtype=dataset[0].dtype
        )

    if test_splits:
        test_array = np.concatenate(test_splits, axis=0).reshape(
            -1, arr.shape[1], arr.shape[2]
        )
    else:
        test_array = np.empty(
            (0, dataset[0].shape[1], dataset[0].shape[2]), dtype=dataset[0].dtype
        )

    return train_array, test_array


def split_random(dataset, test_ratio, seed=0):
    """
    Random sampling split: for each file, randomly permute frames,
    then split at (1 - test_ratio).
    """
    rng = np.random.default_rng(seed)
    train_splits = []
    test_splits = []
    for arr in dataset:
        n = arr.shape[0]
        perm = rng.permutation(n)
        split_idx = int((1 - test_ratio) * n)
        train_splits.append(arr[perm[:split_idx]])
        test_splits.append(arr[perm[split_idx:]])
    return np.stack(train_splits, axis=0), np.stack(test_splits, axis=0)


def split_end(dataset, test_ratio):
    """
    End-of-file split: for each file in dataset, take the last test_ratio
    fraction of frames as the test set, and the preceding frames as the train set.
    """
    train_splits = []
    test_splits = []
    for idx, arr in enumerate(dataset):
        n = arr.shape[0]
        test_count = int(test_ratio * n)
        # ensure at least one frame in each split if possible
        test_count = max(test_count, 1) if n > 1 else 0
        train_count = n - test_count

        print(f"Dataset[{idx}]: Total = {n}, train= {train_count}, test= {test_count}")

        train_splits.append(arr[: n - test_count])
        test_splits.append(arr[n - test_count :])

    # find the minimal lengths so we can stack into a dense ndarray
    min_train = min(a.shape[0] for a in train_splits)
    min_test = min(a.shape[0] for a in test_splits)

    train_array = np.stack([a[:min_train] for a in train_splits], axis=0)
    test_array = np.stack([a[-min_test:] for a in test_splits], axis=0)

    return train_array, test_array


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, cfg.INPUT_DIR)
    output_dir = os.path.join(base_dir, cfg.OUTPUT_DIR)

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    pc_size = cfg.PC_SIZE
    test_ratio = cfg.TEST_RATIO
    shift_arr = np.array(cfg.MMWAVE_RADAR_LOC, dtype=np.float32)
    cfg_obj = PointCloudProcessCFG()

    bin_files = sorted(
        [
            Path(f)
            for f in glob.glob(os.path.join(input_dir, "**", "*.bin"), recursive=True)
        ]
    )

    if not bin_files:
        print(f"ERROR: No .bin files found in {input_dir}")
        return

    # Print configuration info
    print("Configuration:")
    print(f"  PC_SIZE: {pc_size}")
    print(f"  TEST_RATIO: {test_ratio}")
    print(f"  SPLIT_METHOD: {cfg.SPLIT_METHOD}")
    print(f"  FLIP_X: {cfg.FLIP_X}")
    print(f"  FLIP_Y: {cfg.FLIP_Y}")

    dataset = []
    # Parallel processing of .bin files
    with ProcessPoolExecutor() as executor:
        future_to_idx = {
            executor.submit(process_bin_file, fp, pc_size, shift_arr, cfg_obj): i
            for i, fp in enumerate(bin_files)
        }

        # preallocate a list of the right length
        dataset = [None] * len(bin_files)

        for future in tqdm(
            as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Processing .bin files",
            unit="file",
        ):
            idx = future_to_idx[future]
            try:
                dataset[idx] = future.result()
            except Exception as e:
                print(f"WARNING: Skipping {bin_files[idx]} due to error: {e}")

    if not dataset:
        print("ERROR: No data processed; aborting.")
        return

    min_frames = min(arr.shape[0] for arr in dataset)
    train_count = int((1.0 - test_ratio) * min_frames)

    print(f"Minimum frames across all files: {min_frames}")
    print(f"Train count: {train_count}, Test count: {min_frames - train_count}")
    print(f"Total files processed: {len(dataset)}")
    print(f"Point cloud size: {pc_size}")

    # Split according to chosen method
    if cfg.SPLIT_METHOD == "sequential":
        train, test = split_sequential(dataset, train_count, min_frames)
    elif cfg.SPLIT_METHOD == "random":
        train, test = split_random(dataset, test_ratio, seed=cfg.SPLIT_SEED)
    elif cfg.SPLIT_METHOD == "end":
        train, test = split_end(dataset, test_ratio)

    train_path = os.path.join(output_dir, "train.dat")
    test_path = os.path.join(output_dir, "test.dat")
    train.dump(train_path)
    print(f"Saved train.dat {train.shape} to {train_path}")
    test.dump(test_path)
    print(f"Saved test.dat {test.shape} to {test_path}")

    # Create visualizations
    if hasattr(cfg, "CREATE_VISUALIZATIONS") and cfg.CREATE_VISUALIZATIONS:
        frame_rate = getattr(cfg, "VIS_FRAME_RATE", 10)
        max_frames_for_videos = 100
        create_dataset_visualizations(
            train, test, output_dir, frame_rate, max_frames_for_videos
        )
    else:
        print("\nTo enable visualizations, add to configuration.py:")
        print("CREATE_VISUALIZATIONS = True")
        print("VIS_FRAME_RATE = 10 ")


if __name__ == "__main__":
    main()
