import os
import time

import numpy as np
import torch
import torch.nn as nn

import network as mmwave_model

import dataloader as mmwave_data
from smpl_utils_extend import SMPL


class mmwave():
    def __init__(self, exp_name, net, model_batch):
        self.batch_size = 32
        self.train_length = 64
        self.dist_th = 0.6
        self.gpu_id = 0

        self.exp_name = exp_name
        self.net = net
        self.model_batch = model_batch

        self.device = 'cpu'
        self.pc_size = 64

        self.dataset = mmwave_data.data(self.batch_size, self.train_length, self.pc_size)
        self.model = mmwave_model.mmWaveModel(device=self.device).to(self.device)
        self.name_size = 1
        self.act_size = self.dataset.act_size
        self.joint_size = self.dataset.joint_size
        self.test_length_size = self.dataset.test_len
        self.cos = nn.CosineSimilarity(-1)
        self.male_smpl = SMPL('m', device=self.device)
        self.female_smpl = SMPL('f', device=self.device)
        root_kp = np.asarray([17, 19, 16, 18, 2, 5, 1, 4], dtype=np.int64)
        leaf_kp = np.asarray([19, 21, 18, 20, 5, 8, 4, 7], dtype=np.int64)
        self.root_kp = torch.tensor(root_kp, dtype=torch.long, device=self.device)
        self.leaf_kp = torch.tensor(leaf_kp, dtype=torch.long, device=self.device)

    def save_model(self, name):
        torch.save(self.model.state_dict(), './results/' + self.exp_name + '/%s/model/' % self.net + name + '.pth')

    def load_model(self, name):
        self.model.load_state_dict(torch.load('./results/' + self.exp_name + '/%s/model/' % self.net + name + '.pth', map_location=self.device))

    def cal_vs_from_qtbg(self, pquat_tensor, trans_tensor, betas_tensor, gender_tensor, b_size, l_size):
        with torch.no_grad():
            vertice_tensor = torch.zeros((b_size, l_size, 6890, 3), dtype=torch.float32, requires_grad=False, device=self.device)
            ske_tensor = torch.zeros((b_size, l_size, 24, 3), dtype=torch.float32, requires_grad=False, device=self.device)
            pquat_arr = torch.zeros((b_size, l_size, 24, 3, 3), dtype=torch.float32, requires_grad=False, device=self.device)

            wrot_tensor = pquat_tensor[:, 0:1, :, :]
            rotmat_tensor = torch.squeeze(wrot_tensor)

            pquat_arr[:, :, :] = torch.eye(3, dtype=torch.float32, requires_grad=False, device=self.device)
            pquat_arr[:, :, range(self.joint_size)[1:]] = pquat_tensor[:, 1:]  # except root joint

            male_flag = gender_tensor[:, 0, 0] > 0.5
            female_flag = gender_tensor[:, 0, 0] < 0.5
            if male_flag.any().item():
                vertice_tensor[male_flag], ske_tensor[male_flag] = self.male_smpl(betas_tensor[male_flag], pquat_arr[male_flag], torch.zeros((male_flag.sum().item(), l_size, 3), dtype=torch.float32, requires_grad=False, device=self.device))
            if female_flag.any().item():
                vertice_tensor[female_flag], ske_tensor[female_flag] = self.female_smpl(betas_tensor[female_flag], pquat_arr[female_flag], torch.zeros((female_flag.sum().item(), l_size, 3), dtype=torch.float32, requires_grad=False, device=self.device))

            rotmat_tensor = rotmat_tensor.view(b_size * l_size, 3, 3)
            vertice_tensor = vertice_tensor.view(b_size * l_size, 6890, 3)
            ske_tensor = ske_tensor.view(b_size * l_size, 24, 3)
            trans_tensor = trans_tensor.view(b_size * l_size, 1, 3)

            vertice_tensor = torch.transpose(torch.bmm(rotmat_tensor, torch.transpose(vertice_tensor, 1, 2)), 1, 2) + trans_tensor
            ske_tensor = torch.transpose(torch.bmm(rotmat_tensor, torch.transpose(ske_tensor, 1, 2)), 1, 2) + trans_tensor
            vertice_tensor = vertice_tensor.view(b_size, l_size, 6890, 3)
            ske_tensor = ske_tensor.view(b_size, l_size, 24, 3)

        return vertice_tensor.detach(), ske_tensor.detach()

    def angle_loss(self, pred_ske, true_ske):
        pred_vec = pred_ske[:, :, self.leaf_kp, :] - pred_ske[:, :, self.root_kp, :]
        true_vec = true_ske[:, :, self.leaf_kp, :] - true_ske[:, :, self.root_kp, :]
        cos_sim = nn.functional.cosine_similarity(pred_vec, true_vec, dim=-1)
        angle = torch.sum(torch.abs(torch.acos(torch.clamp(cos_sim, min=-1.0, max=1.0)) / 3.14159265358 * 180.0))
        return angle

    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist

    def compute_mpjpe(self, pred_s, target):
        # [B, T, 24, 3]
        mpjpe = torch.sqrt(torch.sum(torch.square(pred_s - target), dim=-1))
        return mpjpe

    def compute_pelvis_mpjpe(self, pred_s, target):
        # [B, T, 24, 3]
        pred_pel = pred_s[:, :, 0:1, :]
        pred = pred_s - pred_pel
        target_pel = target[:, :, 0:1, :]
        target = target - target_pel
        pel_mpjpe = torch.sqrt(torch.sum(torch.square(pred - target), dim=-1))
        return pel_mpjpe

    def compute_pck_b(self, pred_s, target):
        pel_mpjpe = self.compute_pelvis_mpjpe(pred_s, target)
        th = 0.2 * torch.sum(torch.square(target[:, :, 12:13, :] - target[:, :, 0:1, :]), dim=-1)
        pck = torch.count_nonzero(pel_mpjpe < th)
        pck = pck / self.dataset.test_len / self.dataset.joint_size
        return pck

    def infer_all_0(self):
        self.model.eval()

        q_list = []
        t_list = []
        v_list = []
        s_list = []
        l_list = []
        b_list = []
        g_list = []
        pckh_list = []
        pckh2_list = []
        pckb_list = []

        angle_report = 0.0
        trans_report = 0.0
        vertice_report = 0.0
        Pel_MPJPE = 0.0
        ske_report = 0.0
        loc_report = 0.0
        betas_report = 0.0
        gender_acc = 0.0

        np_pc = self.dataset.test_pc
        np_pquat = self.dataset.test_pquat
        np_trans = self.dataset.test_trans

        for data in range(self.dataset.data_len):
            betas_tensor = torch.tensor(np.expand_dims(self.dataset.betas[data: data + 1], 0), dtype=torch.float32, device=self.device)
            gender_tensor = torch.tensor(np.expand_dims(self.dataset.gender[data: data + 1], 0), dtype=torch.float32, device=self.device)

            pc_tensor = torch.tensor([np_pc[data]], dtype=torch.float32, device=self.device)
            pquat_tensor = torch.tensor((np_pquat[data]), dtype=torch.float32, device=self.device)
            trans_tensor = torch.tensor((np_trans[data]), dtype=torch.float32, device=self.device)
            vertice_tensor, ske_tensor = self.cal_vs_from_qtbg(pquat_tensor, trans_tensor, betas_tensor.repeat(1, self.test_length_size, 1), gender_tensor.repeat(1, self.test_length_size, 1), 1, self.test_length_size)

            h0_g = torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
            c0_g = torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
            h0_a = torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)
            c0_a = torch.zeros((3, 1, 64), dtype=torch.float32, device=self.device)

            pred_q, pred_t, pred_v, pred_s, pred_l, pred_b, pred_g, _, _, _, _, _, _ = self.model(pc_tensor, None, h0_g, c0_g, h0_a, c0_a, self.joint_size)
            pred_g[pred_g > 0.5] = 1.0
            pred_g[pred_g <= 0.5] = 0.0
            gender_acc += 1.0 - torch.sum(torch.abs(pred_g - gender_tensor)).item() / self.test_length_size

            pckh = 0
            rate = 0.5
            pckh2 = 0
            rate2 = 1
            h = self.square_distance(ske_tensor[0, :, 15:16, :], ske_tensor[0, :, 12:13, :])
            for f in range(pred_s.shape[1]):
                for joint in range(pred_s.shape[2]):
                    dis = self.square_distance(pred_s[0, f:f + 1, joint:joint + 1, :], ske_tensor[0, f:f + 1, joint:joint + 1, :])[0, 0, 0]
                    if dis < h[f, 0, 0] * rate:
                        pckh += 1
                    if dis < h[f] * rate2:
                        pckh2 = pckh2 + 1
            pckh = pckh / pred_s.shape[2] / pred_s.shape[1]
            pckh_list.append(pckh)
            pckh2 = pckh2 / pred_s.shape[2] / pred_s.shape[1]
            pckh2_list.append(pckh2)

            pckb = self.compute_pck_b(pred_s, ske_tensor).item()
            pckb_list.append(pckb)

            angle_report += self.angle_loss(pred_s, ske_tensor)
            # Pel-MPJPE
            Pel_MPJPE += torch.sum(self.compute_pelvis_mpjpe(pred_s, ske_tensor)).item()
            # MPJPE
            ske_report += torch.sum(torch.sqrt(torch.sum(torch.square(pred_s - ske_tensor), dim=-1))).item()
            vertice_report += torch.sum(torch.sqrt(torch.sum(torch.square(pred_v - vertice_tensor), dim=-1))).item()
            trans_report += torch.sum(torch.sqrt(torch.sum(torch.square(pred_t - trans_tensor), dim=-1))).item()
            loc_report += torch.sum(torch.sqrt(torch.sum(torch.square(pred_l - trans_tensor[..., :2]), dim=-1))).item()
            betas_report += torch.sum(self.cos(pred_b, betas_tensor)).item()
            pred_l = pred_l.cpu().detach().numpy()

            pred_q = pred_q.cpu().detach().numpy()
            pred_t = pred_t.cpu().detach().numpy()
            pred_v = pred_v.cpu().detach().numpy()
            pred_s = pred_s.cpu().detach().numpy()
            pred_b = pred_b.cpu().detach().numpy()
            pred_g = pred_g.cpu().detach().numpy()

            q_list.append(pred_q)
            t_list.append(pred_t)
            v_list.append(pred_v)
            s_list.append(pred_s)
            l_list.append(pred_l)
            b_list.append(pred_b)
            g_list.append(pred_g)
        # return np.array(pc_denoise_list), np.asarray(q_list), np.asarray(t_list), np.asarray(v_list), np.asarray(s_list), np.asarray(l_list), np.asarray(b_list), np.asarray(g_list), np_pc
        angle_report /= (self.dataset.data_len * self.test_length_size * 8)
        trans_report /= (self.dataset.data_len * self.test_length_size)
        vertice_report /= (self.dataset.data_len * self.test_length_size * 6890)
        Pel_MPJPE /= (self.dataset.data_len * self.test_length_size * 24)
        ske_report /= (self.dataset.data_len * self.test_length_size * 24)
        loc_report /= (self.dataset.data_len * self.test_length_size)
        betas_report /= (self.dataset.data_len * self.test_length_size)
        gender_acc /= (self.dataset.data_len)
        return np.asarray(q_list), np.asarray(t_list), np.asarray(v_list), np.asarray(s_list), np.asarray(l_list), np.asarray(b_list), np.asarray(g_list), np_pc,\
                angle_report, trans_report, vertice_report, ske_report, loc_report, betas_report, gender_acc, pckh_list, pckh2_list, pckb_list, Pel_MPJPE


if __name__ == '__main__':
    exp = "20-05"
    net_name = "base"
    batch = "batch80000"

    path = "results/" + exp + "/"  # place results under this directory
    m = mmwave(exp, net_name, batch)
    m.load_model(batch)  # saved model ckeckpoint
    f = open(os.path.join(path, net_name, "metrics.txt"), "a")

    q, t, v, s, l, b, g, pc, angle_r, trans_r, vertice_r, ske_r, loc_r, betas_r, gender_acc, pckh, pckh2, pckb, Pel_MPJPE = m.infer_all_0()
    f.write("pckh@0.5: " + str(pckh) + "\n")
    f.write("pckh@0.5: " + str(sum(pckh) / m.dataset.data_len) + "\n")
    f.write("pckh@1: " + str(pckh2) + "\n")
    f.write("pckh@1: " + str(sum(pckh2) / m.dataset.data_len) + "\n")
    f.write("pckb" + str(pckb) + "\n")
    f.write("pckb" + str(sum(pckb) / m.dataset.data_len) + "\n")
    f.write("Q" + str(angle_r) + "\n")
    f.write("Pel_MPJPE" + str(Pel_MPJPE) + "\n")
    f.write("MPJPE" + str(ske_r) + "\n")
    f.write("vertice" + str(vertice_r) + "\n")
    f.write("trans" + str(trans_r) + "\n")
    f.write("loc" + str(loc_r) + "\n")
    f.write("betas acc" + str(betas_r) + "\n")
    f.write("gender acc" + str(gender_acc) + "\n")

    f.close()
    with open('./%s/%s/%s.pmat.dat' % (path, net_name, net_name), 'wb') as outfile:
        np.save(outfile, q)
    with open('./%s/%s/%s.trans.dat' % (path, net_name, net_name), 'wb') as outfile:
        np.save(outfile, t)
    with open('./%s/%s/%s.vertices.dat' % (path, net_name, net_name), 'wb') as outfile:
        np.save(outfile, v)
    with open('./%s/%s/%s.skeleton.dat' % (path, net_name, net_name), 'wb') as outfile:
        np.save(outfile, s)
    with open('./%s/%s/%s.delta.dat' % (path, net_name, net_name), 'wb') as outfile:
        np.save(outfile, l)
    with open('./%s/%s/%s.beta.dat' % (path, net_name, net_name), 'wb') as outfile:
        np.save(outfile, b)
    with open('./%s/%s/%s.gender.dat' % (path, net_name, net_name), 'wb') as outfile:
        np.save(outfile, g)
    with open('./%s/%s/%s.test_pc.dat' % (path, net_name, net_name), 'wb') as outfile:
        np.save(outfile, pc)
    m.dataset.close()
    print("Done!")
