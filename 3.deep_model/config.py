import torch

CONFIG = {
    "write_slot": 1000,
    "save_slot": 1000,
    "log_slot": 100,
    "visual_slot": 10000,
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
    "mocap_fps": 10,
    "mmwave_fps": 10,
    "split_ratio": 0.1,
    "split_method": "end",  # Options: 'end', 'sequential'
    "pre-visualization": False,
}

DATABASE_CONFIG = {
    "input_path": "data/input",
    "output_path": "data/output",
}
