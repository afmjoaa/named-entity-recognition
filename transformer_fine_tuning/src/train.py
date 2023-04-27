import os

from src.tf_dataset import TfDataset
from src.utility import Constants
from src.model import Model


class Train:
    @staticmethod
    def startTraining():
        os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
        os.environ["TF_GPU_THREAD_COUNT"] = "16"

        # Instantiating tensorflow dataset class
        tf_dataset_class = TfDataset("../data/all_data.json")

        # Instantiating Model class
        model_class = Model(train_ds_count=len(tf_dataset_class.train_ds))
        model = model_class.model

        # Creating tf_dataset
        tf_dataset_class.forward(model=model)

        # Compile model
        model.compile(optimizer=model_class.optimizer)

        # Accessing final datasets
        train_set = tf_dataset_class.final_train_set
        validation_set = tf_dataset_class.final_validation_set

        # Creating callback array
        callbacks = [
            Model.getMetricCallback(validation_set),
            Model.getTensorBoardCallback(),
            Model.getModelCheckPointCallback(),
            Model.getEarlyStoppingCallback(),
        ]

        model.fit(
            train_set,
            validation_data=validation_set,
            epochs=Constants.MAX_EPOCH,
            callbacks=callbacks,
        )
