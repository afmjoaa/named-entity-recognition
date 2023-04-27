import os

from src.label_process import LabelProcess
from src.tf_dataset import TfDataset
from src.utility import Constants
from src.model import Model
import numpy as np
from transformers.keras_callbacks import KerasMetricCallback
from datasets import load_metric
from transformers.keras_callbacks import PushToHubCallback

class Train:
    labelProcess = LabelProcess(labelFileName="../data/label_info.json")

    @staticmethod
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        print(f"predictions {predictions[:20]}")
        print(f"labels {labels[:20]}")

        # Remove ignored index (special tokens)
        true_predictions = [
            [Train.labelProcess.unique_labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [Train.labelProcess.unique_labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        print(f"true_predictions first 10 {true_predictions[:10]}")
        print(f"true_labels first 10 {true_labels[:10]}")

        print(f"true_predictions last 10 {true_predictions[-10:]}")
        print(f"true_labels last 10 {true_labels[-10:]}")

        results = load_metric("seqeval").compute(predictions=true_predictions, references=true_labels, zero_division=0)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    @staticmethod
    def startTraining():
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
        os.environ["TF_GPU_THREAD_COUNT"] = "8"

        # Instantiating tensorflow dataset class
        tf_dataset_class = TfDataset("../data/all_data.json")

        # Instantiating Model class
        model_class = Model(train_ds_count=len(tf_dataset_class.tokenized_train_ds))
        model = model_class.model

        # Creating tf_dataset
        tf_dataset_class.forward(model=model)

        # Compile model
        model.compile(optimizer=model_class.optimizer)

        # Accessing final datasets
        train_set = tf_dataset_class.final_train_set
        validation_set = tf_dataset_class.final_validation_set

        metric_callback = KerasMetricCallback(
            metric_fn=Train.compute_metrics, eval_dataset=validation_set
        )

        # model_name = Constants.MODEL_CHECKPOINT.split("/")[-1]
        # push_to_hub_model_id = f"{model_name}-finetuned-{Constants.TASK}"
        #
        # push_to_hub_callback = PushToHubCallback(
        #     output_dir="./tc_model_save",
        #     tokenizer=model_class.tokenizer,
        #     hub_model_id=push_to_hub_model_id,
        # )

        # Creating callback array
        callbacks = [
            metric_callback,
            Model.getTensorBoardCallback(),
            # Model.getModelCheckPointCallback(),
            Model.getEarlyStoppingCallback(),
            # push_to_hub_callback
        ]

        model.fit(
            train_set,
            validation_data=validation_set,
            epochs=Constants.MAX_EPOCH,
            callbacks=callbacks,
        )
