from pathlib import Path
import torch
import torch.nn as nn
import torch.functional as F

import architecture
import training_loop
import data_generator

# TODO: Add tensorboard or weights and biases monitoring.

class AlexNet:
    def __init__(self, train_directory, val_directory, label_file, batch_size=256) -> None:
        self.model = architecture.AlexNetPytorch()

        self.train_generator= data_generator.create_alexnet_dataloader(
            image_directory=train_directory,
            label_file=label_file,
            batch_size=batch_size
        )
        self.val_generator = data_generator.create_alexnet_dataloader(
            image_directory=val_directory,
            label_file=label_file,
            batch_size=batch_size
        )
        self.optimizer=None
        self.loss_function=None


    def fit(self, epochs):
        training_loop.training_loop(
            epochs=epochs,
            model=self.model,
            optimizer=self.optimizer,
            loss_fn=self.loss_function,
            train_loader=self.train_generator,
            val_loader=self.val_generator,
            device=None
        )

    def save_model(self, save_path: Path):
        torch.save(self.model, save_path)
