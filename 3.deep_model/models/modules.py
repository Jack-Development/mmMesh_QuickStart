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

    def forward(self, q, t, b, g, joint_size):
        # q: (B, T, J, 3, 3), t: (B, T, 1, 3), b: (B, T, 10), g: (B, T)
        B, T = q.size(0), q.size(1)
        N = B * T
        # flatten batch/time dims
        q = q.view(N, joint_size, 3, 3)
        t = t.view(N, 1, 3)
        b = b.view(N, 10)
        g = g.view(N)

        # build the SMPL pose sequence
        q_blank = self.blank_atom.unsqueeze(0).expand(N, -1, -1, -1)  # (N,1,3,3)
        pose = torch.cat(
            [
                q_blank,  # root
                q[:, 1:3],  # some limbs
                q_blank,  # filler
                q[:, 3:5],  # more limbs
                q_blank.expand(N, 10, 3, 3),  # torso/hips
                q[:, 5:9],  # arms/legs
                q_blank.expand(N, 4, 3, 3),  # hands/feet
            ],
            dim=1,
        )  # (N, 24, 3, 3)

        rotmat = q[:, 0]  # (N, 3, 3)

        # split indices
        male_idx = torch.nonzero(g > 0.5, as_tuple=True)[0]
        female_idx = torch.nonzero(g <= 0.5, as_tuple=True)[0]

        # run SMPL for each group
        v_m, s_m = (
            self.smpl_model_m(
                b[male_idx],
                pose[male_idx],
                torch.zeros((male_idx.size(0), 3), device=self.device),
            )
            if male_idx.numel()
            else (
                torch.empty(0, 6890, 3, device=self.device),
                torch.empty(0, 24, 3, device=self.device),
            )
        )

        v_f, s_f = (
            self.smpl_model_f(
                b[female_idx],
                pose[female_idx],
                torch.zeros((female_idx.size(0), 3), device=self.device),
            )
            if female_idx.numel()
            else (
                torch.empty(0, 6890, 3, device=self.device),
                torch.empty(0, 24, 3, device=self.device),
            )
        )

        # concatenate predictions
        verts_all = torch.cat([v_m, v_f], dim=0)  # (Nm+Nf, 6890,3)
        skel_all = torch.cat([s_m, s_f], dim=0)  # (Nm+Nf, 24, 3)

        # build an inverse index to scatter back in original order
        inv_idx = torch.empty(N, dtype=torch.long, device=self.device)
        inv_idx[male_idx] = torch.arange(male_idx.size(0), device=self.device)
        inv_idx[female_idx] = torch.arange(
            female_idx.size(0), device=self.device
        ) + male_idx.size(0)

        # gather back
        smpl_vertice = verts_all[inv_idx]  # (N, 6890,3)
        smpl_skeleton = skel_all[inv_idx]  # (N,  24,3)

        # apply global rotation + translation
        smpl_vertice = (
            torch.bmm(rotmat, smpl_vertice.transpose(1, 2)).transpose(1, 2) + t
        )
        smpl_skeleton = (
            torch.bmm(rotmat, smpl_skeleton.transpose(1, 2)).transpose(1, 2) + t
        )

        # reshape to (B, T, …)
        smpl_vertice = smpl_vertice.view(B, T, 6890, 3)
        smpl_skeleton = smpl_skeleton.view(B, T, 24, 3)

        return smpl_vertice, smpl_skeleton
