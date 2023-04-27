from transformers import AutoTokenizer
from transformers import TFAutoModelForTokenClassification

from src.label_process import LabelProcess
from src.utility import Constants
from transformers import create_optimizer
import tensorflow as tf


class Model:
    def __init__(self, train_ds_count):
        labelProcess = LabelProcess(labelFileName="../data/label_info.json")
        self.model = TFAutoModelForTokenClassification.from_pretrained(
            Constants.MODEL_CHECKPOINT,
            num_labels=len(labelProcess.unique_labels),
            id2label=labelProcess.id2label,
            label2id=labelProcess.label2id,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(Constants.MODEL_CHECKPOINT)
        num_train_epochs = Constants.MAX_EPOCH
        num_train_steps = (train_ds_count // Constants.BATCH_SIZE) * num_train_epochs
        self.optimizer, self.lr_schedule = create_optimizer(
            init_lr=2e-5,
            num_train_steps=num_train_steps,
            weight_decay_rate=0.01,
            num_warmup_steps=0,
        )

    @staticmethod
    def getModelCheckPointCallback():
        return tf.keras.callbacks.ModelCheckpoint(
            "../model/model_best",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
        )

    @staticmethod
    def getTensorBoardCallback():
        return tf.keras.callbacks.TensorBoard(log_dir="../logs")

    @staticmethod
    def getEarlyStoppingCallback():
        return tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, mode="min", verbose=1
        )
