import torch

CONFIG = {
    "write_slot": 1000,
    "save_slot": 1000,
    "log_slot": 100,
    "batch_size": 32,
    "batch_rate": 1,
    "train_size": 80_000,
    "train_length": 64,
    "learning_rate": 0.001,
    "gpu_id": 0,
    "pc_size": 128,
    "train_eval_size": 10,
    "vertice_rate": 0.001,
    "betas_rate": 0.1,
    "device": f"{'cuda:0' if torch.cuda.is_available() else 'cpu'}",
}

DATABASE_CONFIG = {
    "input_path": "data/input",
    "output_path": "data/output",
}
