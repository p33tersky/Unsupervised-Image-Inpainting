import comet_ml
from lightning.pytorch.loggers import CometLogger

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from inpainting import LitModel
from dataset import train_loader, test_loader

EXPERIMENT_NAME = "inpainting training"

comet_logger = CometLogger(
    api_key= "0RfBkzG1OWWq2b8BwzxjFp0Q2",
    project_name="...",
    experiment_name=EXPERIMENT_NAME,
)

best_checkpoint = ModelCheckpoint(
    monitor='validation_loss',
    dirpath=f'checkpoints/{EXPERIMENT_NAME}/',
    filename='model-{epoch:02d}-{validation_loss:.2f}',
    save_top_k=1,
    mode='min'
)
# model-epoch=14-validation_loss=0.2

last_checkpoint = ModelCheckpoint(
    dirpath=f'checkpoints/{EXPERIMENT_NAME}/',
    filename='model-{epoch:02d}',
    save_top_k=1,
    every_n_epochs=1,
)
# model-epoch=20

early_stopping = EarlyStopping(
    monitor='validation_loss',
    patience=5,
    mode='min'
)

trainer = pl.Trainer(
    max_epochs=30,
    callbacks=[last_checkpoint, best_checkpoint, early_stopping],
    logger=comet_logger
)

lit_model = LitModel()
trainer.fit(lit_model, train_loader, test_loader)