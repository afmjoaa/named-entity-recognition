from datasets import load_metric
from transformers import AutoTokenizer
from transformers import TFAutoModelForTokenClassification

from src.label_process import LabelProcess
from src.utility import Constants
from transformers import create_optimizer
import numpy as np
from transformers.keras_callbacks import KerasMetricCallback
import tensorflow as tf


class Model:
    metric = load_metric("seqeval")
    labelProcess = LabelProcess(labelFileName="../data/label_info.json")

    def __init__(self, train_ds_count):
        self.model = TFAutoModelForTokenClassification.from_pretrained(
            Constants.MODEL_CHECKPOINT,
            num_labels=len(Model.labelProcess.unique_labels),
            id2label=Model.labelProcess.id2label,
            label2id=Model.labelProcess.label2id,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(Constants.MODEL_CHECKPOINT)
        num_train_epochs = Constants.MAX_EPOCH
        num_train_steps = (train_ds_count // Constants.BATCH_SIZE) * num_train_epochs
        self.optimizer, self.lr_schedule = create_optimizer(
            init_lr=2e-2,
            num_train_steps=num_train_steps,
            weight_decay_rate=0.01,
            num_warmup_steps=0,
        )

    @staticmethod
    def _compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        unique_labels = Model.labelProcess.unique_labels
        # Remove ignored index (special tokens)
        true_predictions = [
            [unique_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [unique_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = Model.metric.compute(
            predictions=true_predictions, references=true_labels
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    @staticmethod
    def getMetricCallback(validation_set):
        return KerasMetricCallback(
            metric_fn=Model._compute_metrics, eval_dataset=validation_set
        )

    @staticmethod
    def getModelCheckPointCallback():
        return tf.keras.callbacks.ModelCheckpoint(
            "../model/model_best.h5",
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
