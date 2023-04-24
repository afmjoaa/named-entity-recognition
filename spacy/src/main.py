from spacy_dataset import SpacyDataset
from train import Train
from inference import Inference


def CreateSpacyDataset():
    train_file_name = "../../data/en-train.conll"
    dev_file_name = "../../data/en-train.conll"
    spacyDataset = SpacyDataset(train_file_name, dev_file_name)
    (
        train_formatted_array,
        dev_formatted_array,
        test_formatted_array,
    ) = spacyDataset.init_data_split()
    print(train_formatted_array[0])
    print(dev_formatted_array[0])
    print(test_formatted_array[0])
    spacyDataset.saveSpacyDataSet()


if __name__ == "__main__":
    # For creating dataset uncomment the CreateSpacyDataset function (line 23)
    # CreateSpacyDataset()

    # For training the model uncomment line 25
    # Train().startTraining(isResumed=False)

    # For evaluating the trained model use line 30
    Inference().evaluateTestDataset(None)
