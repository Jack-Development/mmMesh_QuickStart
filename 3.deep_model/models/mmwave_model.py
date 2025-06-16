import torch
import torch.nn as nn
from .networks import BasePointNet
from .modules import (
    GlobalModule,
    AnchorModule,
    CombineModule,
    SMPLModule,
)


class mmWaveModel(nn.Module):
    def __init__(self, joint_size=22):
        super(mmWaveModel, self).__init__()
        self.module0 = BasePointNet()
        self.module1 = GlobalModule()
        self.module2 = AnchorModule()
        self.module3 = CombineModule(joint_size)
        self.module4 = SMPLModule()

    def forward(self, x, gt_g, h0_g, c0_g, h0_a, c0_a, joint_size):
        batch_size = x.size()[0]
        length_size = x.size()[1]
        pt_size = x.size()[2]
        in_feature_size = x.size()[3]
        out_feature_size = 24 + 4

        x = x.view(batch_size * length_size, pt_size, in_feature_size)
        x = self.module0(x)

        g_vec, g_loc, global_weights, hn_g, cn_g = self.module1(
            x, h0_g, c0_g, batch_size, length_size
        )
        a_vec, anchor_weights, hn_a, cn_a = self.module2(
            x, g_loc, h0_a, c0_a, batch_size, length_size, out_feature_size
        )
        q, t, b, g = self.module3(g_vec, a_vec, batch_size, length_size, joint_size)

        if gt_g is None:
            g_in = torch.round(g)
        else:
            g_in = gt_g
        v, s = self.module4(q, t, b, g_in, joint_size)
        return (
            q,
            t,
            v,
            s,
            g_loc,
            b,
            g,
            global_weights,
            anchor_weights,
            hn_g,
            cn_g,
            hn_a,
            cn_a,
        )
