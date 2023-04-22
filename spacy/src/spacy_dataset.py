import random
from preprocess import PreProcess
import spacy
from spacy.tokens import DocBin
import os


class SpacyDataset:
    def __init__(self, train_file_location, dev_file_location):
        self.train_file_location = train_file_location
        self.dev_file_location = dev_file_location
        self.raw_data_array = []
        self.train_formatted_array = []
        self.test_formatted_array = []
        self.dev_formatted_array = []

    def init_data_split(self):
        train_file_data = PreProcess.getDataArrayAsMap(self.train_file_location)
        dev_file_data = PreProcess.getDataArrayAsMap(self.train_file_location)
        all_data_list = train_file_data + dev_file_data

        random.shuffle(all_data_list)
        self.raw_data_array = all_data_list

        n_train = int(len(self.raw_data_array) * 0.7)
        n_dev = int(len(self.raw_data_array) * 0.2)

        train_raw_array = self.raw_data_array[:n_train]
        dev_raw_array = self.raw_data_array[n_train : n_train + n_dev]
        test_raw_array = self.raw_data_array[n_train + n_dev :]

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

    def saveSpacyDataSet(self):
        nlp = spacy.blank("en")
        train_doc_bin = DocBin()
        dev_doc_bin = DocBin()
        test_doc_bin = DocBin()

        train_doc_array = list(
            map(
                lambda x: PreProcess.createDocFromFormattedMap(x, nlp),
                self.train_formatted_array,
            )
        )
        dev_doc_array = list(
            map(
                lambda x: PreProcess.createDocFromFormattedMap(x, nlp),
                self.dev_formatted_array,
            )
        )
        test_doc_array = list(
            map(
                lambda x: PreProcess.createDocFromFormattedMap(x, nlp),
                self.test_formatted_array,
            )
        )

        for item in train_doc_array:
            train_doc_bin.add(item)

        for item in dev_doc_array:
            dev_doc_bin.add(item)

        for item in test_doc_array:
            test_doc_bin.add(item)

        directory = "../data"
        if not os.path.exists(directory):
            os.mkdir(directory)

        train_doc_bin.to_disk("../data/training_data.spacy")
        dev_doc_bin.to_disk("../data/dev_data.spacy")
        test_doc_bin.to_disk("../data/test_data.spacy")
