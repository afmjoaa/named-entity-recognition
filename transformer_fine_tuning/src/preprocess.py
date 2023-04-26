"""
Create raw .json file will all data available.
"""
import random
import json
from utility import Constants, Utility


class PreProcess:
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
                elif line.strip() == "":
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
    def saveRawDataInJson(filename):
        train_file_location = "../../data/en-train.conll"
        dev_file_location = "../../data/en-dev.conll"
        train_file_data = PreProcess.getDataArrayAsMap(train_file_location)
        dev_file_data = PreProcess.getDataArrayAsMap(dev_file_location)
        all_data_list = train_file_data + dev_file_data
        random.shuffle(all_data_list)

        formatted_data_list = list(map(PreProcess.addSentenceAndLabelKey, all_data_list))

        with open(filename, 'w') as f:
            json.dump(formatted_data_list, f)

    @staticmethod
    def addSentenceAndLabelKey(sample):
        sentence = ""
        label_arr = []
        for word, label in sample.items():
            label_arr.append(label)
            sentence += word + " "
        return {"sentence": sentence.strip(), "label_array": label_arr}



