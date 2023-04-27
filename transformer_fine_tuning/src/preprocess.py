"""
Create raw .json file will all data available.
Create raw .json file for label information.
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
    def getLabelArray(dataFile: str, onlyBioTagging=False):
        label_array = []
        with open(dataFile) as f:
            for line in f:
                if line.startswith(Constants.ID_IDENTIFIER):
                    pass
                elif line.strip() == "":
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
    def saveLabelInfo(fileName):
        train_file_location = "../../data/en-train.conll"
        dev_file_location = "../../data/en-dev.conll"
        train_file_label = PreProcess.getLabelArray(train_file_location)
        dev_file_label = PreProcess.getLabelArray(dev_file_location)
        all_label_list = train_file_label + dev_file_label

        unique_labels = Utility.get_unique_items(all_label_list)
        unique_labels.sort(reverse=True)
        id2label = {i: label for i, label in enumerate(unique_labels)}
        label2id = {label: i for i, label in enumerate(unique_labels)}

        data = {
            "unique_labels": unique_labels,
            "id2label": id2label,
            "label2id": label2id,
        }

        with open(fileName, "w") as f:
            json.dump(data, f)

    @staticmethod
    def readLabelInfo(fileName, destructure):
        with open(fileName, "r") as f:
            data = json.load(f)

        unique_labels = data["unique_labels"]
        id2label = data["id2label"]
        label2id = data["label2id"]

        if destructure:
            return unique_labels, id2label, label2id
        else:
            return data

    @staticmethod
    def addSentenceAndLabelKey(sample, label2id):
        sentence = ""
        ner_tags = []
        for word, label in sample.items():
            ner_tags.append(label2id[label])
            sentence += word + " "
        return {"sentence": sentence.strip(), "ner_tags": ner_tags}

    @staticmethod
    def saveRawDataInJson(dataFileName, labelFileName):
        train_file_location = "../../data/en-train.conll"
        dev_file_location = "../../data/en-dev.conll"
        train_file_data = PreProcess.getDataArrayAsMap(train_file_location)
        dev_file_data = PreProcess.getDataArrayAsMap(dev_file_location)
        all_data_list = train_file_data + dev_file_data
        random.shuffle(all_data_list)

        _, __, label2id = PreProcess.readLabelInfo(labelFileName, True)

        formatted_data_list = list(
            map(lambda x: PreProcess.addSentenceAndLabelKey(x, label2id), all_data_list)
        )

        with open(dataFileName, "w") as f:
            json.dump(formatted_data_list, f)
