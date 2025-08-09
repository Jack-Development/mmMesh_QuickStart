from train.trainer import Trainer
from config import CONFIG

if __name__ == "__main__":
    trainer = Trainer()
    cfg = CONFIG
    if cfg["pre-visualization"]:
        trainer.verify_before_training()
    trainer.train_model()
    trainer.dataset.close()
