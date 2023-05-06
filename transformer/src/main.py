import src.train as t
from src.transformer_dataset import TransformerDataset
import train as Train
def CreateTransformerDataset():
    train_file_name = "../../data/en-train.conll"
    dev_file_name = "../../data/en-dev.conll"

    formatted_train_file = "../data/train_data.json"
    formatted_dev_file = "../data/dev_data.json"
    formatted_test_file = "../data/test_data.json"

    tranDataset = TransformerDataset(
        train_file_name,
        dev_file_name,
        formatted_train_file,
        formatted_dev_file,
        formatted_test_file)
    allSentenceArray = tranDataset.getAllSentenceArray()
    print(len(allSentenceArray))
    (
        train_formatted_array,
        dev_formatted_array,
        test_formatted_array,
    ) = tranDataset.divideData()
    print(len(train_formatted_array))
    print(len(dev_formatted_array))
    print(len(test_formatted_array))
    print(train_formatted_array[0])

    # create data.json file and inserts formatted data
    tranDataset.storeData()

    # read data from data.json file
    print(tranDataset.readData(train=True)[0])


if __name__ == "__main__":
    # For creating dataset uncomment the CreateTransformerDataset function (line 29)
    CreateTransformerDataset()

    # For training the model uncomment line 25
    # Train.startTraining()

    # For evaluating the trained model use line 30
    # Inference().evaluateTestDataset(None)