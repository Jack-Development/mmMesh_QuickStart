import sys
import os

import numpy as np
import torch
import pickle as pk

import models.mmwave_model as mmwave_model

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../2.point_cloud_generation")
    )
)
import configuration as cfg
from pc_generation import (
    PointCloudProcessCFG,
    RawDataReader,
    bin2np_frame,
    frame2pointcloud,
    reg_data,
)

FLIP_X = cfg.FLIP_X
FLIP_Y = cfg.FLIP_Y


class mmwave:
    def __init__(self, model_path, pc_size):
        self.model_path = model_path
        self.pc_size = pc_size
        self.joint_size = 22

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = mmwave_model.mmWaveModel().to(self.device)
        self.load_model(model_path)

    def load_model(self, model_path):
        print(f"[mmWave] Loading model from {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    def load_data(self, name):
        pointCloudProcessCFG = PointCloudProcessCFG()
        shift_arr = cfg.MMWAVE_RADAR_LOC
        bin_reader = RawDataReader(name)
        total_frames = bin_reader.getTotalFrames(pointCloudProcessCFG.frameConfig)
        data = np.zeros((total_frames, self.pc_size, 6))

        for frame_no in range(total_frames):
            bin_frame = bin_reader.getNextFrame(pointCloudProcessCFG.frameConfig)
            np_frame = bin2np_frame(bin_frame)
            pointCloud = frame2pointcloud(np_frame, pointCloudProcessCFG)
            if pointCloud.shape[0] == 0 or pointCloud.shape[1] == 0:
                continue

            raw_points = np.transpose(pointCloud, (1, 0))
            raw_points[:, :3] = raw_points[:, :3] + shift_arr
            raw_points = apply_coordinate_flips(raw_points)
            raw_points = reg_data(raw_points, self.pc_size)

            data[frame_no] = raw_points

        bin_reader.close()
        print(
            f"[mmWave] Loaded data from {name}, total frames: {total_frames}, point cloud size: {self.pc_size}"
        )
        return data

    def infer(self, filename):
        self.model.eval()

        np_pc = self.load_data(filename)
        np_pc = np.array([np_pc])

        pc_tensor = torch.tensor(np_pc, dtype=torch.float32, device=self.device)

        h0_g = torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
        c0_g = torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
        h0_a = torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
        c0_a = torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)

        pred_q, pred_t, pred_v, pred_s, pred_l, pred_b, pred_g, _, _, _, _, _, _ = (
            self.model(pc_tensor, None, h0_g, c0_g, h0_a, c0_a, self.joint_size)
        )
        pred_g[pred_g > 0.5] = 1.0
        pred_g[pred_g <= 0.5] = 0.0

        pred_l = pred_l.cpu().detach().numpy().squeeze()
        pred_q = pred_q.cpu().detach().numpy().squeeze()
        pred_t = pred_t.cpu().detach().numpy().squeeze()
        pred_v = pred_v.cpu().detach().numpy().squeeze()
        pred_s = pred_s.cpu().detach().numpy().squeeze()
        pred_b = pred_b.cpu().detach().numpy().squeeze()
        pred_g = pred_g.cpu().detach().numpy().squeeze()

        return pred_q, pred_t, pred_v, pred_s, pred_l, pred_b, pred_g, np_pc


def apply_coordinate_flips(points):
    if FLIP_X:
        points[:, 0] = -points[:, 0]  # Flip X coordinate

    if FLIP_Y:
        points[:, 1] = -points[:, 1]  # Flip Y coordinate

    return points


if __name__ == "__main__":
    exp = "13-05"
    net_name = "base"
    batch = "batch80000"

    path = "results/" + exp + "/"
    m = mmwave(path, 128)
    m.load_model(batch)  # saved model ckeckpoint
    q, t, v, s, l, b, g, pc = m.infer(
        "/home/jack/ドキュメント/002/CODE/PnP_002/mmWave/input/mmWave.bin"
    )

    print("s shape:", s.shape)
    with open(path + "s.pkl", "wb") as f:
        pk.dump(s, f)
