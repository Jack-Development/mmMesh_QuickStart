from train.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_model()
    trainer.dataset.close()
