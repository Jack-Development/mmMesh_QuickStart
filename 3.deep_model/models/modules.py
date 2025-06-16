import torch.nn as nn
import torch
from .utils import AnchorInit, AnchorGrouping
from .networks import (
    AnchorPointNet,
    AnchorVoxelNet,
    AnchorRNN,
    GlobalPointNet,
    GlobalRNN,
)
from smpl_models.smpl_wrapper import SMPLWrapper


class AnchorModule(nn.Module):
    def __init__(self):
        super(AnchorModule, self).__init__()
        self.template_point = AnchorInit()
        self.z_size, self.y_size, self.x_size, _ = self.template_point.shape
        self.anchor_size = self.z_size * self.y_size * self.x_size
        self.apointnet = AnchorPointNet()
        self.avoxel = AnchorVoxelNet()
        self.arnn = AnchorRNN()

    def forward(self, x, g_loc, h0, c0, batch_size, length_size, feature_size):
        g_loc = g_loc.view(batch_size * length_size, 1, 2).repeat(
            1, self.anchor_size, 1
        )
        anchors = self.template_point.view(1, self.anchor_size, 3).repeat(
            batch_size * length_size, 1, 1
        )
        anchors[:, :, :2] += g_loc
        grouped_points = AnchorGrouping(
            anchors, nsample=8, xyz=x[..., :3], points=x[..., 3:]
        )
        grouped_points = grouped_points.view(
            batch_size * length_size * self.anchor_size, 8, 3 + feature_size
        )
        voxel_points, attn_weights = self.apointnet(grouped_points)
        voxel_points = voxel_points.view(
            batch_size * length_size, self.z_size, self.y_size, self.x_size, 64
        )
        voxel_vec = self.avoxel(voxel_points)
        voxel_vec = voxel_vec.view(batch_size, length_size, 64)
        a_vec, hn, cn = self.arnn(voxel_vec, h0, c0)
        return a_vec, attn_weights, hn, cn


class GlobalModule(nn.Module):
    def __init__(self):
        super(GlobalModule, self).__init__()
        self.gpointnet = GlobalPointNet()
        self.grnn = GlobalRNN()

    def forward(self, x, h0, c0, batch_size, length_size):
        x, attn_weights = self.gpointnet(x)
        x = x.view(batch_size, length_size, 64)
        g_vec, g_loc, hn, cn = self.grnn(x, h0, c0)
        return g_vec, g_loc, attn_weights, hn, cn


class CombineModule(nn.Module):
    def __init__(self, joint_size=22):
        super(CombineModule, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, joint_size * 6 + 3 + 10 + 1)

    def forward(self, g_vec, a_vec, batch_size, length_size, joint_size=22):
        x = torch.cat((g_vec, a_vec), -1)
        x = self.fc1(x)
        x = self.faf1(x)
        x = self.fc2(x)

        q = (
            x[:, :, : joint_size * 6]
            .reshape(batch_size * length_size * joint_size, 6)
            .contiguous()
        )
        tmp_x = nn.functional.normalize(q[:, :3], dim=-1)
        tmp_z = nn.functional.normalize(torch.cross(tmp_x, q[:, 3:], dim=-1), dim=-1)
        tmp_y = torch.cross(tmp_z, tmp_x, dim=-1)

        tmp_x = tmp_x.view(batch_size, length_size, joint_size, 3, 1)
        tmp_y = tmp_y.view(batch_size, length_size, joint_size, 3, 1)
        tmp_z = tmp_z.view(batch_size, length_size, joint_size, 3, 1)
        q = torch.cat((tmp_x, tmp_y, tmp_z), -1)

        t = x[:, :, joint_size * 6 : joint_size * 6 + 3]
        b = x[:, :, joint_size * 6 + 3 : joint_size * 6 + 3 + 10]
        g = x[:, :, joint_size * 6 + 3 + 10 :]
        return q, t, b, g


class SMPLModule(nn.Module):
    def __init__(self):
        super(SMPLModule, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.blank_atom = torch.tensor(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            dtype=torch.float32,
            requires_grad=False,
            device=self.device,
        )
        self.smpl_wrapper = SMPLWrapper()
        self.smpl_model_m = self.smpl_wrapper.male_smpl
        self.smpl_model_f = self.smpl_wrapper.female_smpl

    def forward(self, q, t, b, g, joint_size):  # b: (10,)
        batch_size = q.size()[0]
        length_size = q.size()[1]
        q = q.view(batch_size * length_size, joint_size, 3, 3)
        t = t.view(batch_size * length_size, 1, 3)
        b = b.view(batch_size * length_size, 10)
        g = g.view(batch_size * length_size)
        q_blank = self.blank_atom.repeat(batch_size * length_size, 1, 1, 1)
        pose = torch.cat(
            (
                q_blank,
                q[:, 1:3, :, :],
                q_blank,
                q[:, 3:5, :, :],
                q_blank.repeat(1, 10, 1, 1),
                q[:, 5:9, :, :],
                q_blank.repeat(1, 4, 1, 1),
            ),
            1,
        )
        rotmat = q[:, 0, :, :]

        male = g > 0.5
        female = g < 0.5
        smpl_vertice = torch.zeros(
            (batch_size * length_size, 6890, 3),
            dtype=torch.float32,
            requires_grad=False,
            device=self.device,
        )
        smpl_skeleton = torch.zeros(
            (batch_size * length_size, 24, 3),
            dtype=torch.float32,
            requires_grad=False,
            device=self.device,
        )
        if male.any().item():
            smpl_vertice[male], smpl_skeleton[male] = self.smpl_model_m(
                b[male],
                pose[male],
                torch.zeros(
                    (male.sum().item(), 3),
                    dtype=torch.float32,
                    requires_grad=False,
                    device=self.device,
                ),
            )
        if female.any().item():
            smpl_vertice[female], smpl_skeleton[female] = self.smpl_model_f(
                b[female],
                pose[female],
                torch.zeros(
                    (female.sum().item(), 3),
                    dtype=torch.float32,
                    requires_grad=False,
                    device=self.device,
                ),
            )

        smpl_vertice = (
            torch.transpose(
                torch.bmm(rotmat, torch.transpose(smpl_vertice, 1, 2)), 1, 2
            )
            + t
        )
        smpl_skeleton = (
            torch.transpose(
                torch.bmm(rotmat, torch.transpose(smpl_skeleton, 1, 2)), 1, 2
            )
            + t
        )
        smpl_vertice = smpl_vertice.view(batch_size, length_size, 6890, 3)
        smpl_skeleton = smpl_skeleton.view(batch_size, length_size, 24, 3)
        return smpl_vertice, smpl_skeleton
