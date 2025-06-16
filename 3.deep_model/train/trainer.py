import os
import time
from os.path import join
from glob import glob

import torch
from torch.utils.tensorboard import SummaryWriter

from config import CONFIG, DATABASE_CONFIG
from data_loader.dataloader import DataLoader
from models.mmwave_model import mmWaveModel
from smpl_models.smpl_wrapper import SMPLWrapper
from train.utils import hinge_loss
from train.evaluator import Evaluator


class Trainer:
    def __init__(self, base_dir=None):
        if base_dir is None:
            # Get the directory of the current file and go up one level
            base_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(base_dir)

        self.data_cfg = DATABASE_CONFIG
        self.input_path = os.path.join(base_dir, self.data_cfg["input_path"])
        self.output_path = os.path.join(base_dir, self.data_cfg["output_path"])

        os.makedirs(self.input_path, exist_ok=True)
        os.makedirs(self.output_path, exist_ok=True)

        self.cfg = CONFIG
        self.configure()

        mocap_path = os.path.join(self.input_path, "mocap_data")
        os.makedirs(mocap_path, exist_ok=True)
        mmwave_path = os.path.join(self.input_path, "mmwave_data")
        os.makedirs(mmwave_path, exist_ok=True)

        mocap_paths = glob(
            os.path.join(self.input_path, "mocap_data", "*.pkl"),
            recursive=True,
        )

        mmwave_train = os.path.join(self.input_path, "mmwave_data", "train.dat")
        mmwave_test = os.path.join(self.input_path, "mmwave_data", "test.dat")

        self.dataset = DataLoader(
            mocap_paths=mocap_paths,
            mmwave_train_path=mmwave_train,
            mmwave_test_path=mmwave_test,
            batch_size=self.cfg["batch_size"],
            seq_length=self.cfg["train_length"],
            pc_size=self.cfg["pc_size"],
            test_split_ratio=0.2,
            prefetch_size=128,
            test_buffer=2,
            num_workers=4,
            device=self.device,
        )

        self.model = mmWaveModel(self.dataset.joint_size).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.cfg["learning_rate"]
        )
        self.criterion = torch.nn.L1Loss(reduction="sum")
        self.criterion_gender = hinge_loss
        self.cos = torch.nn.CosineSimilarity(-1)

        self.smpl = SMPLWrapper()

        # Init constant tensors
        self.root_kp = torch.tensor(
            [17, 19, 16, 18, 2, 5, 1, 4], dtype=torch.long, device=self.device
        )
        self.leaf_kp = torch.tensor(
            [19, 21, 18, 20, 5, 8, 4, 7], dtype=torch.long, device=self.device
        )

    def configure(self):
        self.write_slot = self.cfg["write_slot"]
        self.save_slot = self.cfg["save_slot"]
        self.log_slot = self.cfg["log_slot"]
        self.batch_size = self.cfg["batch_size"]
        self.batch_rate = self.cfg["batch_rate"]
        self.train_size = self.cfg["train_size"]
        self.train_length = self.cfg["train_length"]
        self.learning_rate = self.cfg["learning_rate"]
        self.gpu_id = self.cfg["gpu_id"]
        self.pc_size = self.cfg["pc_size"]
        self.train_eval_size = self.cfg["train_eval_size"]
        self.vertice_rate = self.cfg["vertice_rate"]
        self.betas_rate = self.cfg["betas_rate"]
        self.device = self.cfg["device"]

        print(
            "Trainer configured with batch_size=%d, train_size=%d, train_length=%d, device=%s"
            % (
                self.batch_size,
                self.train_size,
                self.train_length,
                self.device,
            )
        )

    def save_model(self, name):
        save_path = join(self.output_path, "checkpoints", "model")
        os.makedirs(save_path, exist_ok=True)
        torch.save(self.model.state_dict(), join(save_path, name + ".pth"))

    def load_model(self, name):
        load_path = join(self.output_path, "checkpoints", "model")
        os.makedirs(load_path, exist_ok=True)
        self.model.load_state_dict(
            torch.load(join(load_path, name + ".pth"), map_location=self.device)
        )

    def cal_vs_from_qtbg(
        self, pquat_tensor, trans_tensor, betas_tensor, gender_tensor, b_size, l_size
    ):
        with torch.no_grad():
            vertice_tensor = torch.zeros(
                (b_size, l_size, 6890, 3),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )
            ske_tensor = torch.zeros(
                (b_size, l_size, 24, 3),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )
            wrot_tensor = pquat_tensor[:, :, 0:1, :, :]
            rotmat_tensor = torch.squeeze(wrot_tensor)
            pquat_arr = torch.zeros(
                (b_size, l_size, 24, 3, 3),
                dtype=torch.float32,
                requires_grad=False,
                device=self.device,
            )
            pquat_arr[:, :, :] = torch.eye(
                3, dtype=torch.float32, requires_grad=False, device=self.device
            )
            pquat_arr[:, :, range(self.dataset.joint_size)[1:]] = pquat_tensor[:, :, 1:]
            male_flag = gender_tensor[:, 0, 0] > 0.5
            female_flag = gender_tensor[:, 0, 0] < 0.5
            if male_flag.any().item():
                vertice_tensor[male_flag], ske_tensor[male_flag] = self.smpl.male_smpl(
                    betas_tensor[male_flag],
                    pquat_arr[male_flag],
                    torch.zeros(
                        (male_flag.sum().item(), l_size, 3),
                        dtype=torch.float32,
                        requires_grad=False,
                        device=self.device,
                    ),
                )
            if female_flag.any().item():
                vertice_tensor[female_flag], ske_tensor[female_flag] = (
                    self.smpl.female_smpl(
                        betas_tensor[female_flag],
                        pquat_arr[female_flag],
                        torch.zeros(
                            (female_flag.sum().item(), l_size, 3),
                            dtype=torch.float32,
                            requires_grad=False,
                            device=self.device,
                        ),
                    )
                )

            rotmat_tensor = rotmat_tensor.view(b_size * l_size, 3, 3)
            vertice_tensor = vertice_tensor.view(b_size * l_size, 6890, 3)
            ske_tensor = ske_tensor.view(b_size * l_size, 24, 3)
            trans_tensor = trans_tensor.view(b_size * l_size, 1, 3)

            vertice_tensor = (
                torch.transpose(
                    torch.bmm(rotmat_tensor, torch.transpose(vertice_tensor, 1, 2)),
                    1,
                    2,
                )
                + trans_tensor
            )
            ske_tensor = (
                torch.transpose(
                    torch.bmm(rotmat_tensor, torch.transpose(ske_tensor, 1, 2)), 1, 2
                )
                + trans_tensor
            )
            vertice_tensor = vertice_tensor.view(b_size, l_size, 6890, 3)
            ske_tensor = ske_tensor.view(b_size, l_size, 24, 3)
        return vertice_tensor.detach(), ske_tensor.detach()

    def train_once(self):
        self.model.train()

        self.model.zero_grad()

        for i in range(self.batch_rate):
            h0_g = torch.zeros(
                (3, self.batch_size, 64), dtype=torch.float32, device=self.device
            )
            c0_g = torch.zeros(
                (3, self.batch_size, 64), dtype=torch.float32, device=self.device
            )
            h0_a = torch.zeros(
                (3, self.batch_size, 64), dtype=torch.float32, device=self.device
            )
            c0_a = torch.zeros(
                (3, self.batch_size, 64), dtype=torch.float32, device=self.device
            )

            pc, pquat, trans, betas, gender = self.dataset.next_batch()
            pc_tensor = torch.tensor(pc, dtype=torch.float32, device=self.device)
            pquat_tensor = torch.tensor(pquat, dtype=torch.float32, device=self.device)
            trans_tensor = torch.tensor(trans, dtype=torch.float32, device=self.device)
            betas_tensor = torch.tensor(betas, dtype=torch.float32, device=self.device)
            gender_tensor = torch.tensor(
                gender, dtype=torch.float32, device=self.device
            )
            vertice_tensor, ske_tensor = self.cal_vs_from_qtbg(
                pquat_tensor,
                trans_tensor,
                betas_tensor,
                gender_tensor,
                self.batch_size,
                self.train_length,
            )

            pred_q, pred_t, pred_v, pred_s, pred_l, pred_b, pred_g, _, _, _, _, _, _ = (
                self.model(
                    pc_tensor,
                    gender_tensor,
                    h0_g,
                    c0_g,
                    h0_a,
                    c0_a,
                    self.dataset.joint_size,
                )
            )

            loss = (
                self.vertice_rate * self.criterion(pred_v, vertice_tensor)
                + self.criterion(pred_s, ske_tensor)
                + self.criterion(pred_l, trans_tensor[..., :2])
                + self.betas_rate * self.criterion(pred_b, betas_tensor)
                + self.criterion_gender(pred_g, gender_tensor)
            )
            loss.backward()

        self.optimizer.step()
        return

    def train_model(self):
        evaluator = Evaluator(self)
        timestamp = time.strftime("%Y%m%d_%H%M")
        loss_path = join(self.output_path, "loss", timestamp + ".txt")
        eval_path = join(self.output_path, "eval", timestamp + ".txt")
        log_dir = join(self.output_path, "logs", timestamp)

        os.makedirs(join(self.output_path, "loss"), exist_ok=True)
        os.makedirs(join(self.output_path, "eval"), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        def clear_terminal():
            os.system("cls" if os.name == "nt" else "clear")

        # Initial evaluation and logging
        metrics = evaluator.evaluate()
        (
            train_pquat_loss,
            train_trans_loss,
            train_vertice_loss,
            train_ske_loss,
            train_loc_loss,
            train_betas_loss,
            train_gender_loss,
            train_angle_report,
            train_trans_report,
            train_vertice_report,
            train_ske_report,
            train_loc_report,
            train_betas_report,
            pquat_loss,
            trans_loss,
            vertice_loss,
            ske_loss,
            loc_loss,
            betas_loss,
            gender_loss,
            angle_report,
            trans_report,
            vertice_report,
            ske_report,
            loc_report,
            betas_report,
            gender_acc,
        ) = metrics

        with open(loss_path, "w") as lossfile, open(eval_path, "w") as evalfile:
            # log to tensorboard at step 0
            writer.add_scalar("train/pquat_loss", train_pquat_loss, 0)
            writer.add_scalar("train/trans_loss", train_trans_loss, 0)
            writer.add_scalar("train/vertice_loss", train_vertice_loss, 0)
            writer.add_scalar("train/ske_loss", train_ske_loss, 0)
            writer.add_scalar("train/loc_loss", train_loc_loss, 0)
            writer.add_scalar("train/betas_loss", train_betas_loss, 0)
            writer.add_scalar("train/gender_loss", train_gender_loss, 0)
            writer.add_scalar("eval/pquat_loss", pquat_loss, 0)
            writer.add_scalar("eval/trans_loss", trans_loss, 0)
            writer.add_scalar("eval/vertice_loss", vertice_loss, 0)
            writer.add_scalar("eval/ske_loss", ske_loss, 0)
            writer.add_scalar("eval/loc_loss", loc_loss, 0)
            writer.add_scalar("eval/betas_loss", betas_loss, 0)
            writer.add_scalar("eval/gender_loss", gender_loss, 0)

            # write to files
            lossfile.write(
                f"0 {train_pquat_loss} {train_trans_loss} {train_vertice_loss} {train_ske_loss} {train_loc_loss} {train_betas_loss} {train_gender_loss}\n"
            )
            evalfile.write(
                f"0 {pquat_loss} {trans_loss} {vertice_loss} {ske_loss} {loc_loss} {betas_loss} {gender_loss}\n"
            )
            lossfile.flush()
            evalfile.flush()

            # Print initial metrics
            clear_terminal()
            print("Iteration: 0")
            print("-" * 40)
            print("Train Losses:")
            print(f"  pquat: {train_pquat_loss:.3f}")
            print(f"  trans: {train_trans_loss:.3f}")
            print(f"  vertice: {train_vertice_loss:.3f}")
            print(f"  ske: {train_ske_loss:.3f}")
            print(f"  loc: {train_loc_loss:.3f}")
            print(f"  betas: {train_betas_loss:.3f}")
            print(f"  gender: {train_gender_loss:.3f}")
            print("-" * 40)
            print("Eval Losses:")
            print(f"  pquat: {pquat_loss:.3f}")
            print(f"  trans: {trans_loss:.3f}")
            print(f"  vertice: {vertice_loss:.3f}")
            print(f"  ske: {ske_loss:.3f}")
            print(f"  loc: {loc_loss:.3f}")
            print(f"  betas: {betas_loss:.3f}")
            print(f"  gender: {gender_loss:.3f}")
            print("-" * 40)

            begin_time = time.time()
            for i in range(self.train_size):
                self.train_once()
                step = i + 1
                need_write = step % self.write_slot == 0
                need_print = need_write or (step < 1000 and step % 100 == 0)
                need_save = step % self.save_slot == 0
                need_log = step % self.log_slot == 0

                if need_write or need_print or need_log:
                    metrics = evaluator.evaluate()
                    (
                        train_pquat_loss,
                        train_trans_loss,
                        train_vertice_loss,
                        train_ske_loss,
                        train_loc_loss,
                        train_betas_loss,
                        train_gender_loss,
                        train_angle_report,
                        train_trans_report,
                        train_vertice_report,
                        train_ske_report,
                        train_loc_report,
                        train_betas_report,
                        pquat_loss,
                        trans_loss,
                        vertice_loss,
                        ske_loss,
                        loc_loss,
                        betas_loss,
                        gender_loss,
                        angle_report,
                        trans_report,
                        vertice_report,
                        ske_report,
                        loc_report,
                        betas_report,
                        gender_acc,
                    ) = metrics

                if need_log:
                    # tensorboard logging
                    writer.add_scalar("train/pquat_loss", train_pquat_loss, step)
                    writer.add_scalar("train/trans_loss", train_trans_loss, step)
                    writer.add_scalar("train/vertice_loss", train_vertice_loss, step)
                    writer.add_scalar("train/ske_loss", train_ske_loss, step)
                    writer.add_scalar("train/loc_loss", train_loc_loss, step)
                    writer.add_scalar("train/betas_loss", train_betas_loss, step)
                    writer.add_scalar("train/gender_loss", train_gender_loss, step)
                    writer.add_scalar("eval/pquat_loss", pquat_loss, step)
                    writer.add_scalar("eval/trans_loss", trans_loss, step)
                    writer.add_scalar("eval/vertice_loss", vertice_loss, step)
                    writer.add_scalar("eval/ske_loss", ske_loss, step)
                    writer.add_scalar("eval/loc_loss", loc_loss, step)
                    writer.add_scalar("eval/betas_loss", betas_loss, step)
                    writer.add_scalar("eval/gender_loss", gender_loss, step)

                if need_write:
                    lossfile.write(
                        f"{step} {train_pquat_loss} {train_trans_loss} {train_vertice_loss} {train_ske_loss} {train_loc_loss} {train_betas_loss} {train_gender_loss} \n"
                    )
                    evalfile.write(
                        f"{step} {pquat_loss} {trans_loss} {vertice_loss} {ske_loss} {loc_loss} {betas_loss} {gender_loss}\n"
                    )
                    lossfile.flush()
                    evalfile.flush()

                if need_print:
                    elapsed = (time.time() - begin_time) / 3600
                    eta = elapsed / step * self.train_size

                    remaining = max(0.0, eta - elapsed)
                    finish_ts = time.time() + remaining * 3600
                    finish_struct = time.localtime(finish_ts)
                    now_struct = time.localtime()
                    if (finish_struct.tm_year, finish_struct.tm_yday) != (
                        now_struct.tm_year,
                        now_struct.tm_yday,
                    ):
                        finish_str = time.strftime("%Y-%m-%d %H:%M:%S", finish_struct)
                    else:
                        finish_str = time.strftime("%H:%M:%S", finish_struct)

                    clear_terminal()
                    print(f"Iteration: {step}")
                    print("-" * 40)
                    print("Train Losses:")
                    print(f"  pquat: {train_pquat_loss:.3f}")
                    print(f"  trans: {train_trans_loss:.3f}")
                    print(f"  vertice: {train_vertice_loss:.3f}")
                    print(f"  ske: {train_ske_loss:.3f}")
                    print(f"  loc: {train_loc_loss:.3f}")
                    print(f"  betas: {train_betas_loss:.3f}")
                    print(f"  gender: {train_gender_loss:.3f}")
                    print("-" * 40)
                    print("Eval Losses:")
                    print(f"  pquat: {pquat_loss:.3f}")
                    print(f"  trans: {trans_loss:.3f}")
                    print(f"  vertice: {vertice_loss:.3f}")
                    print(f"  ske: {ske_loss:.3f}")
                    print(f"  loc: {loc_loss:.3f}")
                    print(f"  betas: {betas_loss:.3f}")
                    print(f"  gender: {gender_loss:.3f}")
                    print("-" * 40)
                    print(
                        f"Elapsed: {elapsed:.2f} h | ETA: {eta:.2f} h | Finish at: {finish_str}"
                    )

                if need_save:
                    self.save_model(f"batch{step}")

            writer.close()
