import torch
import numpy as np
from train.utils import hinge_loss


class Evaluator:
    """
    Evaluation logic for training and testing using the new DataLoader.
    """

    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.model
        self.dataset = trainer.dataset
        self.device = trainer.device
        self.vertice_rate = trainer.vertice_rate
        self.betas_rate = trainer.betas_rate
        self.batch_size = trainer.batch_size
        self.train_eval_size = trainer.train_eval_size

        # Use DataLoader's attributes
        self.train_length = self.dataset.seq_length
        self.test_length = self.dataset.test_length
        self.num_samples = self.dataset.num_samples

        # Loss functions
        self.criterion = torch.nn.L1Loss(reduction="sum")
        self.criterion_gender = hinge_loss
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            train_metrics = self._eval_training()
            test_metrics = self._eval_test()
        self.model.train()
        return (*train_metrics, *test_metrics)

    def _eval_training(self):
        # Initialize accumulators
        keys = [
            "pquat_loss",
            "trans_loss",
            "vertice_loss",
            "ske_loss",
            "loc_loss",
            "betas_loss",
            "gender_loss",
        ]
        accum = dict.fromkeys(keys, 0.0)

        # Sample and accumulate
        for _ in range(self.train_eval_size):
            pc, pquat, trans, betas, gender = self.dataset.next_batch()
            metrics = self._compute_metrics(
                pc,
                pquat,
                trans,
                betas,
                gender,
                self.trainer.root_kp,
                self.trainer.leaf_kp,
                is_train=True,
            )
            for k, v in metrics.items():
                accum[k] += v

        # Normalize by total frames evaluated
        norm = self.batch_size * self.train_eval_size * self.train_length
        return [
            accum["pquat_loss"] / norm,
            accum["trans_loss"] / norm,
            accum["vertice_loss"] / norm * self.vertice_rate,
            accum["ske_loss"] / norm,
            accum["loc_loss"] / norm,
            accum["betas_loss"] / norm * self.betas_rate,
            accum["gender_loss"] / norm,
        ]

    def _eval_test(self):
        # Initialize accumulators with gender_acc
        keys = [
            "pquat_loss",
            "trans_loss",
            "vertice_loss",
            "ske_loss",
            "loc_loss",
            "betas_loss",
            "gender_loss",
        ]
        accum = dict.fromkeys(keys, 0.0)

        # Retrieve full test sets
        pc_all = self.dataset.get_test()  # [N, T, P, 6]
        pquat_all = self.dataset.m_pquat_test  # [N, T, J, 3, 3]
        trans_all = self.dataset.m_trans_test  # [N, T, 3]

        # Evaluate each sample as a single sequence
        for i in range(self.num_samples):
            pc = pc_all[i][None, ...]
            pquat = pquat_all[i][None, ...]
            trans = trans_all[i][None, ...]

            # repeat betas/gender over time axis, then batch axis
            betas_base = np.repeat(
                self.dataset.betas[i][None, :], self.test_length, axis=0
            )[None, ...]
            gender_base = np.repeat(
                [self.dataset.gender[i][None]], self.test_length, axis=0
            )[None, ...]

            metrics = self._compute_metrics(
                pc,
                pquat,
                trans,
                betas_base,
                gender_base,
                self.trainer.root_kp,
                self.trainer.leaf_kp,
                is_train=False,
            )
            for k, v in metrics.items():
                accum[k] += v

        # Normalize test metrics per-frame
        norm = self.num_samples * self.test_length
        return [
            accum["pquat_loss"] / norm,
            accum["trans_loss"] / norm,
            accum["vertice_loss"] / norm * self.vertice_rate,
            accum["ske_loss"] / norm,
            accum["loc_loss"] / norm,
            accum["betas_loss"] / norm * self.betas_rate,
            accum["gender_loss"] / norm,
        ]

    def _compute_metrics(
        self,
        pc_np,
        pquat_np,
        trans_np,
        betas_np,
        gender_np,
        root_kp,
        leaf_kp,
        is_train,
    ):
        # Convert to tensors
        bsize = pc_np.shape[0]
        seq_len = self.train_length if is_train else self.test_length
        pc_tensor = torch.tensor(pc_np, dtype=torch.float32, device=self.device)
        pquat_tensor = torch.tensor(pquat_np, dtype=torch.float32, device=self.device)
        trans_tensor = torch.tensor(trans_np, dtype=torch.float32, device=self.device)
        betas_tensor = torch.tensor(betas_np, dtype=torch.float32, device=self.device)
        gender_tensor = torch.tensor(gender_np, dtype=torch.float32, device=self.device)

        # Ground truth vertices/skeletons
        vertice_tensor, ske_tensor = self.trainer.cal_vs_from_qtbg(
            pquat_tensor, trans_tensor, betas_tensor, gender_tensor, bsize, seq_len
        )

        # Initial hidden states
        def h0():
            return torch.zeros((3, bsize, 64), device=self.device)

        h0_g, c0_g, h0_a, c0_a = h0(), h0(), h0(), h0()

        # Model forward
        if is_train:
            inputs = (
                pc_tensor,
                gender_tensor,
                h0_g,
                c0_g,
                h0_a,
                c0_a,
                self.trainer.dataset.joint_size,
            )
        else:
            inputs = (
                pc_tensor,
                None,
                h0_g,
                c0_g,
                h0_a,
                c0_a,
                self.trainer.dataset.joint_size,
            )
        preds = self.model(*inputs)
        pred_q, pred_t, pred_v, pred_s, pred_l, pred_b, pred_g, *_ = preds

        # Compute basic losses
        metrics = {
            "pquat_loss": self.criterion(pred_q, pquat_tensor).item(),
            "trans_loss": self.criterion(pred_t, trans_tensor).item(),
            "vertice_loss": self.criterion(pred_v, vertice_tensor).item(),
            "ske_loss": 5.0 * self.criterion(pred_s, ske_tensor).item(),
            "loc_loss": self.criterion(pred_l, trans_tensor[..., :2]).item(),
            "betas_loss": self.criterion(pred_b, betas_tensor).item(),
            "gender_loss": self.criterion_gender(pred_g, gender_tensor).item(),
        }

        return metrics
