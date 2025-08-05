import sys
import os

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch
import cv2

from smpl_utils_extend import SMPL
import dataloader as mmwave_data


class PlotFigure:
    def __init__(self, dataPath="./"):
        self.batch_size = 32
        self.train_length = 64
        self.pc_size = 128
        self.gpu_id = 0
        # if torch.cuda.is_available():
        #     self.device='cuda:%d'%(self.gpu_id)
        # else:
        self.device = "cpu"

        self.dataPath = dataPath
        self.smpl = [SMPL("f", device=self.device), SMPL("m", device=self.device)]
        self.dataset = mmwave_data.data(
            self.batch_size, self.train_length, self.pc_size
        )
        self.test_length = self.dataset.test_len

    def load_mmMesh_data(self, exp_name, f_name):
        self.out_beta = np.load(
            "./results/%s/%s/%s.beta.dat" % (exp_name, f_name, f_name)
        )
        self.out_delta = np.load(
            "./results/%s/%s/%s.delta.dat" % (exp_name, f_name, f_name)
        )
        self.out_pmat = np.load(
            "./results/%s/%s/%s.pmat.dat" % (exp_name, f_name, f_name)
        )
        self.out_skeleton = np.load(
            "./results/%s/%s/%s.skeleton.dat" % (exp_name, f_name, f_name)
        )
        self.out_test_pc = np.load(
            "./results/%s/%s/%s.test_pc.dat" % (exp_name, f_name, f_name)
        )
        self.out_trans = np.load(
            "./results/%s/%s/%s.trans.dat" % (exp_name, f_name, f_name)
        )
        self.out_vertices = np.load(
            "./results/%s/%s/%s.vertices.dat" % (exp_name, f_name, f_name)
        )
        gender = np.load("./results/%s/%s/%s.gender.dat" % (exp_name, f_name, f_name))
        self.out_gender = np.where(gender < 0.5, 0, 1)

    def load_mocap_data(self):
        _frame = self.dataset.test_pquat.shape[1]
        # Use SMPL to calculate vertices
        _betas = np.tile(self.dataset.betas[:, np.newaxis, :], (1, _frame, 1))
        pquat_tensor = torch.tensor(
            self.dataset.test_pquat, dtype=torch.float32, device=self.device
        )
        trans_tensor = torch.tensor(
            self.dataset.test_trans, dtype=torch.float32, device=self.device
        )
        betas_tensor = torch.tensor(_betas, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            vertice_tensor = torch.zeros(
                (self.dataset.data_len, _frame, 6890, 3),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )
            ske_tensor = torch.zeros(
                (self.dataset.data_len, _frame, 24, 3),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )
            pquat_arr = torch.zeros(
                (self.dataset.data_len, _frame, 24, 3, 3),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )
            pquat_arr[:, :, :, :] = torch.eye(
                3, dtype=torch.float32, requires_grad=False, device=self.device
            )
            pquat_arr[:, :, range(self.dataset.joint_size)[1:]] = pquat_tensor[
                :, :, 1:
            ]  # except root joint
            for gender, data in zip(self.dataset.gender, range(self.dataset.data_len)):
                gender = int(gender)
                vertice_tensor[data], ske_tensor[data] = self.smpl[gender](
                    betas_tensor[data],
                    pquat_arr[data],
                    torch.zeros(
                        (_frame, 3),
                        dtype=torch.float32,
                        requires_grad=False,
                        device=self.device,
                    ),
                )

                wrot_tensor = pquat_tensor[data, :, 0:1, :, :]
                rotmat_tensor = torch.squeeze(wrot_tensor)
                rotmat_tensor = rotmat_tensor.view(_frame, 3, 3)
                v_t = vertice_tensor[data].view(_frame, 6890, 3)
                s_t = ske_tensor[data].view(_frame, 24, 3)
                t_t = trans_tensor[data].view(_frame, 1, 3)

                v_t = (
                    torch.transpose(
                        torch.bmm(rotmat_tensor, torch.transpose(v_t, 1, 2)), 1, 2
                    )
                    + t_t
                )
                s_t = (
                    torch.transpose(
                        torch.bmm(rotmat_tensor, torch.transpose(s_t, 1, 2)), 1, 2
                    )
                    + t_t
                )
                vertice_tensor[data] = v_t.view(_frame, 6890, 3)
                ske_tensor[data] = s_t.view(_frame, 24, 3)

            self.gt_vertices = vertice_tensor.to("cpu").detach().numpy().copy()
            self.gt_skeleton = ske_tensor.to("cpu").detach().numpy().copy()

    def load_mmwave_data(self):
        self.pc = np.array(self.dataset.test_pc)

    def sphere_grid(self, center, r):
        u, v = np.mgrid[0 : 2 * np.pi : 30j, 0 : np.pi : 20j]
        x = np.cos(u) * np.sin(v) * r + center[0]
        y = np.sin(u) * np.sin(v) * r + center[1]
        z = np.cos(v) * r + center[2]

        return x, y, z

    def make_video(self, data):
        del_size = 0
        shape = cv2.imread(savepath + str(data) + "/frame0.png").shape
        videoname = moviepath + str(data) + ".mp4"
        # Video Writer
        fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        video = cv2.VideoWriter(
            videoname, fourcc, 10.0, (shape[1], shape[0] - del_size * 2)
        )
        if not video.isOpened():
            print("can't be opened")
            sys.exit()

        for fr in range(self.test_length):
            figurepath = savepath + str(data) + "/frame" + str(fr) + ".png"
            img = cv2.imread(figurepath)
            img = img[del_size : shape[0] - del_size, :]
            # can't read image, escape
            if img is None:
                print("can't read")
                break

            video.write(img)
            print(fr)

        video.release()
        print("written")

    def plot_all(self, savepath):
        for data in range(self.dataset.data_len):
            for frame in range(self.test_length):
                os.makedirs(savepath + str(data) + "/", exist_ok=True)
                figurepath = savepath + str(data) + "/frame" + str(frame) + ".png"

                plt.rcParams["font.size"] = 3
                fig, axes = plt.subplots(
                    4,  # RGB video, mocap GT, mmWave result
                    3,  # 8
                    sharex="col",
                    sharey=True,
                    subplot_kw=dict(projection="3d"),
                    constrained_layout=True,
                )

                axes[0, 0].set_title("raw point cloud")
                axes[0, 1].set_title("GT pose")
                axes[0, 2].set_title("predicted pose")
                for i in range(4):
                    # plot mmWave result
                    face_color = (0.5, 0.7, 1.0)
                    edge_color = (0.2, 0.2, 0.2)
                    for j in range(3):
                        if i == 0:
                            axes[i, j].set_xlabel("X", size=4)
                            axes[i, j].set_ylabel("Y", size=4)
                            axes[i, j].set_zlabel("Z", size=4)
                            axes[i, j].set_xlim(-2.0, 2.0)
                            axes[i, j].set_ylim(-2.0, 2.0)
                            axes[i, j].set_zlim(-0.5, 2.0)
                            axes[i, j].set_xticks([-2.0, -1.0, 0, 1.0, 2.0])
                            axes[i, j].set_yticks([-2.0, -1.0, 0, 1.0, 2.0])
                            axes[i, j].set_zticks([-0.5, 0, 0.5, 1.0, 1.5, 2.0])
                            axes[i, j].view_init(azim=-75, elev=20)
                            axes[i, j].grid(False)
                        elif i == 1:
                            axes[i, j].set_xlabel("X", size=4)
                            axes[i, j].set_zlabel("Z", size=4)
                            axes[i, j].set_xlim(-2.0, 2.0)
                            axes[i, j].set_ylim(-2.0, 2.0)
                            axes[i, j].set_zlim(-0.5, 2.0)
                            axes[i, j].set_xticks([-2.0, -1.0, 0, 1.0, 2.0])
                            axes[i, j].set_zticks([-0.5, 0, 0.5, 1.0, 1.5, 2.0])
                            axes[i, j].view_init(azim=-90, elev=0)
                            axes[i, j].grid(False)
                        elif i == 2:
                            axes[i, j].set_ylabel("Y", size=4)
                            axes[i, j].set_zlabel("Z", size=4)
                            axes[i, j].set_xlim(-2.0, 2.0)
                            axes[i, j].set_ylim(-2.0, 2.0)
                            axes[i, j].set_zlim(-0.5, 2.0)
                            axes[i, j].set_yticks([-2.0, -1.0, 0, 1.0, 2.0])
                            axes[i, j].set_zticks([-0.5, 0, 0.5, 1.0, 1.5, 2.0])
                            axes[i, j].view_init(azim=0, elev=0)
                            axes[i, j].grid(False)
                        else:
                            axes[i, j].set_xlabel("X", size=4)
                            axes[i, j].set_ylabel("Y", size=4)
                            axes[i, j].set_xlim(-2.0, 2.0)
                            axes[i, j].set_ylim(-2.0, 2.0)
                            axes[i, j].set_zlim(-0.5, 2.0)
                            axes[i, j].set_xticks([-2.0, -1.0, 0, 1.0, 2.0])
                            axes[i, j].set_yticks([-2.0, -1.0, 0, 1.0, 2.0])
                            axes[i, j].view_init(azim=-90, elev=90)
                            axes[i, j].grid(False)

                    # mmwave result
                    axes[i, 0].scatter(
                        self.pc[data, frame, :, 0],
                        self.pc[data, frame, :, 1],
                        self.pc[data, frame, :, 2],
                        s=1,
                    )

                    # mocap data
                    meshData = list(
                        zip(
                            self.gt_vertices[data, frame, :, 0],
                            self.gt_vertices[data, frame, :, 1],
                            self.gt_vertices[data, frame, :, 2],
                        )
                    )
                    faces = self.smpl[round(self.dataset.gender[data])].faces.astype(
                        int
                    )
                    poly3d = [
                        [meshData[faces[ix][iy]] for iy in range(len(faces[0]))]
                        for ix in range(len(faces))
                    ]
                    gt_mesh = Poly3DCollection(
                        poly3d,
                        linewidths=0.1,
                        edgecolors="k",
                        facecolors="w",
                        alpha=0.2,
                    )
                    gt_mesh.set_edgecolor(edge_color)
                    gt_mesh.set_facecolor(face_color)
                    axes[i, 1].add_collection3d(gt_mesh)

                    # mmMesh data
                    meshData = list(
                        zip(
                            self.out_vertices[data, 0, frame, :, 0],
                            self.out_vertices[data, 0, frame, :, 1],
                            self.out_vertices[data, 0, frame, :, 2],
                        )
                    )
                    faces = self.smpl[self.out_gender[data, 0, frame, 0]].faces.astype(
                        int
                    )
                    poly3d = [
                        [meshData[faces[ix][iy]] for iy in range(len(faces[0]))]
                        for ix in range(len(faces))
                    ]
                    gt_mesh = Poly3DCollection(
                        poly3d,
                        linewidths=0.1,
                        edgecolors="k",
                        facecolors="w",
                        alpha=0.2,
                    )
                    gt_mesh.set_edgecolor(edge_color)
                    gt_mesh.set_facecolor(face_color)
                    axes[i, 2].add_collection3d(gt_mesh)

                    # Skeleton data
                    skeletonData = self.out_skeleton[data, 0, frame, :, :].copy()
                    axes[i, 2].scatter(
                        skeletonData[:, 0],
                        skeletonData[:, 1],
                        skeletonData[:, 2],
                        s=10,
                        c="red",
                        alpha=1.0,
                    )

                fig.savefig(figurepath, dpi=1000, bbox_inches="tight", pad_inches=0)
                plt.cla()
                plt.close()
                print(data, frame)

            self.make_video(data)


if __name__ == "__main__":
    exp_name = "20-05_80000_64"

    f_name = "base"

    plot = PlotFigure()
    plot.load_mmMesh_data(exp_name, f_name)
    plot.load_mocap_data()
    plot.load_mmwave_data()

    savepath = "./results/" + exp_name + "/%s/figure/" % (f_name)
    moviepath = "./results/" + exp_name + "/%s/movie/" % (f_name)
    os.makedirs(savepath, exist_ok=True)
    os.makedirs(moviepath, exist_ok=True)

    plot.plot_all(savepath)  # use this if you want to plot mmMesh(base) result only
