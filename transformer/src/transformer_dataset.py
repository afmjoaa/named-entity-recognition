import os

from src.utils import Constants
from src.preprocess import PreProcess
import random
import json

class TransformerDataset:
    def __init__(
            self,
            train_file_location,
            dev_file_location,
            formatted_train_file,
            formatted_dev_file,
            formatted_test_file
    ):
        self.train_file_location = train_file_location
        self.dev_file_location = dev_file_location
        self.formatted_train_file = formatted_train_file
        self.formatted_dev_file = formatted_dev_file
        self.formatted_test_file = formatted_test_file
        self.raw_data_array = []
        self.sentence_array = []
        self.train_formatted_array = []
        self.test_formatted_array = []
        self.dev_formatted_array = []

    def getAllSentenceArray(self):

        # For train file
        self.getSentenceArray(self.train_file_location)

        # For dev file
        self.getSentenceArray(self.dev_file_location)

        return self.sentence_array

    def getSentenceArray(self, location):

        with open(location, encoding='utf-8') as f:
            # Initialize variables
            current_sent_array = ""

            for line in f:
                if line.startswith(Constants.ID_IDENTIFIER):
                    if len(current_sent_array) > 0:
                        self.sentence_array.append(current_sent_array)
                        current_sent_array = ""
                elif line.strip() == "":
                    pass
                else:
                    # Split the line into its constituent parts
                    parts = line.strip().split(Constants.SEPERATOR)
                    # Get the word and label for this token
                    word = parts[0]
                    current_sent_array += word + " "

            # If there are any tokens left in the current sentence, add it to the list of sentences
            if len(current_sent_array) > 0:
                self.sentence_array.append(current_sent_array)


    def divideData(self):
        train_file_data = PreProcess.getDataArrayAsMap(self.train_file_location)
        dev_file_data = PreProcess.getDataArrayAsMap(self.train_file_location)
        all_data_list = train_file_data + dev_file_data

        random.shuffle(all_data_list)
        self.raw_data_array = all_data_list

        n_train = int(len(self.raw_data_array) * 0.7)
        n_dev = int(len(self.raw_data_array) * 0.2)

        train_raw_array = self.raw_data_array[:n_train]
        dev_raw_array = self.raw_data_array[n_train: n_train + n_dev]
        test_raw_array = self.raw_data_array[n_train + n_dev:]

        self.train_formatted_array = list(
            map(PreProcess.formatMapForNer, train_raw_array)
        )
        self.dev_formatted_array = list(map(PreProcess.formatMapForNer, dev_raw_array))
        self.test_formatted_array = list(
            map(PreProcess.formatMapForNer, test_raw_array)
        )

        return (
            self.train_formatted_array,
            self.dev_formatted_array,
            self.test_formatted_array,
        )

    def storeData(self):
        folder_path = "../data"

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # train data
        with open(self.formatted_train_file, 'w', encoding='utf-8') as file:
            # for d in self.train_formatted_array:
            #     f.write(d['sentence'] + '\t' + d['tag'] + '\n')
            json.dump(self.train_formatted_array, file)

        # dev data
        with open(self.formatted_dev_file, 'w', encoding='utf-8') as file:
            json.dump(self.dev_formatted_array, file)

        # test data
        with open(self.formatted_test_file, 'w', encoding='utf-8') as file:
            json.dump(self.test_formatted_array, file)

    def readData(self, train=False):

        if train:
            with open(self.formatted_train_file, "r") as file:
                data = json.load(file)

            with open(self.formatted_dev_file, "r") as file:
                data.append(json.load(file))

        else:
            with open(self.formatted_test_file, "r") as file:
                data = json.load(file)
        # print(data[0])
        return data



