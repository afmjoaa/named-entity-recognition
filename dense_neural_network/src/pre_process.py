from src.util.constants import Constants
from word_to_vector import WordToVector
from one_hot_encoding import OneHotEncoder
import numpy as np

class PreProcess:
    def __init__(self):
        pass

    @staticmethod
    def getDataArrayAsMap(dataFile: str, onlyBioTagging=False):
        # Open the CoNLL file
        data_array = []
        with open(dataFile) as f:
            # Initialize variables
            current_sent_dictionary = {}

            for line in f:
                if line.startswith(Constants.ID_IDENTIFIER):
                    if len(current_sent_dictionary) > 0:
                        data_array.append(current_sent_dictionary)
                        current_sent_dictionary = {}
                elif line.strip() == '':
                    pass
                else:
                    # Split the line into its constituent parts
                    parts = line.strip().split(Constants.SEPERATOR)
                    # Get the word and label for this token
                    word = parts[0]
                    label = parts[-1]
                    if onlyBioTagging:
                        bio_parts = label.strip().split(Constants.BIOX_SEPERATOR)
                        only_bio_tag = bio_parts[0]
                        current_sent_dictionary[word] = only_bio_tag
                    else:
                        current_sent_dictionary[word] = label

            # If there are any tokens left in the current sentence, add it to the list of sentences
            if len(current_sent_dictionary) > 0:
                data_array.append(current_sent_dictionary)

        return data_array

    @staticmethod
    def getSentenceArray(dataFile: str):
        sentence_array = []
        with open(dataFile) as f:
            # Initialize variables
            current_sent_array = []

            for line in f:
                if line.startswith(Constants.ID_IDENTIFIER):
                    if len(current_sent_array) > 0:
                        sentence_array.append(current_sent_array)
                        current_sent_array = []
                elif line.strip() == '':
                    pass
                else:
                    # Split the line into its constituent parts
                    parts = line.strip().split(Constants.SEPERATOR)
                    # Get the word and label for this token
                    word = parts[0]
                    current_sent_array.append(word)

            # If there are any tokens left in the current sentence, add it to the list of sentences
            if len(current_sent_array) > 0:
                sentence_array.append(current_sent_array)

        return sentence_array

    @staticmethod
    def getWordArray(dataFile: str):
        word_array = []
        with open(dataFile) as f:
            for line in f:
                if line.startswith(Constants.ID_IDENTIFIER):
                    pass
                elif line.strip() == '':
                    pass
                else:
                    # Split the line into its constituent parts
                    parts = line.strip().split(Constants.SEPERATOR)
                    # Get the word and label for this token
                    word = parts[0]
                    word_array.append(word)
        return word_array

    @staticmethod
    def getLabelArray(dataFile: str, onlyBioTagging=False):
        label_array = []
        with open(dataFile) as f:
            for line in f:
                if line.startswith(Constants.ID_IDENTIFIER):
                    pass
                elif line.strip() == '':
                    pass
                else:
                    # Split the line into its constituent parts
                    parts = line.strip().split(Constants.SEPERATOR)
                    # Get the word and label for this token
                    label = parts[-1]
                    if onlyBioTagging:
                        bio_parts = label.strip().split(Constants.BIOX_SEPERATOR)
                        only_bio_tag = bio_parts[0]
                        label_array.append(only_bio_tag)
                    else:
                        label_array.append(label)
        return label_array

    @staticmethod
    def getTrainingTuple(dataFile: str, onlyBioTagging=False):
        word_array = []
        label_array = []
        with open(dataFile) as f:
            for line in f:
                if line.startswith(Constants.ID_IDENTIFIER):
                    pass
                elif line.strip() == '':
                    pass
                else:
                    # Split the line into its constituent parts
                    parts = line.strip().split(Constants.SEPERATOR)
                    # Get the word and label for this token
                    word = parts[0]
                    word_array.append(word)
                    label = parts[-1]
                    if onlyBioTagging:
                        bio_parts = label.strip().split(Constants.BIOX_SEPERATOR)
                        only_bio_tag = bio_parts[0]
                        label_array.append(only_bio_tag)
                    else:
                        label_array.append(label)
        return word_array, label_array

    @staticmethod
    def loadAndSPlitDataSet(train_file_location, dev_file_location):
        trainWordArr, trainLabelArr = PreProcess.getTrainingTuple(dataFile=train_file_location, onlyBioTagging=True)
        devWordArr, devLabelArr = PreProcess.getTrainingTuple(dataFile=dev_file_location, onlyBioTagging=True)

        final_train_arr = trainWordArr + devWordArr
        final_label_arr = trainLabelArr + devLabelArr
        print("Train array length: ", len(final_train_arr))
        print("Label array length: ", len(final_label_arr))

        # Get wordToVector from [wordArr] and oneHotEncoding from [labelArr]
        wordToVecArr = WordToVector.getPretrainedWordToVecList(final_train_arr)
        oneHotEncodingArr = OneHotEncoder.getOneHotEncodingOfOutput(final_label_arr)

        # Convert python array to num py array
        np_wordToVecArr = np.array(wordToVecArr)
        np_oneHotEncodingArr = np.array(oneHotEncodingArr)

        # Shuffle the data randomly
        # np.random.shuffle(np_wordToVecArr)
        # np.random.shuffle(np_oneHotEncodingArr)

        # Split the data into train, validation, and test sets
        n = len(np_wordToVecArr)
        train_split = int(0.7 * n)
        val_split = int(0.2 * n)
        train_data = np_wordToVecArr[:train_split]
        val_data = np_wordToVecArr[train_split:train_split + val_split]
        test_data = np_wordToVecArr[train_split + val_split:]

        # Split the data into train, validation, and test sets
        n_label = len(np_oneHotEncodingArr)
        train_label_split = int(0.7 * n_label)
        val_label_split = int(0.2 * n_label)
        train_label = np_wordToVecArr[:train_label_split]
        val_label = np_wordToVecArr[train_label_split:train_label_split + val_label_split]
        test_label = np_wordToVecArr[train_label_split + val_label_split:]

        return train_data, train_label, val_data, val_label, test_data, test_label





