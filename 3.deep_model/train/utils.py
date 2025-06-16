import torch
import torch.nn as nn


def hinge_loss(x, y):
    return torch.sum(nn.ReLU()(y + (1.0 - 2 * y) * x))


def angle_loss(pred_ske, true_ske, root_kp, leaf_kp):
    pred_vec = pred_ske[:, :, leaf_kp, :] - pred_ske[:, :, root_kp, :]
    true_vec = true_ske[:, :, leaf_kp, :] - true_ske[:, :, root_kp, :]
    cos_sim = nn.functional.cosine_similarity(pred_vec, true_vec, dim=-1)
    angle = torch.sum(
        torch.abs(
            torch.acos(torch.clamp(cos_sim, min=-1.0, max=1.0)) / 3.14159265358 * 180.0
        )
    )
    return angle
