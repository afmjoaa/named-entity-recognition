import numpy as np

from src.utility import Constants


class LabelProcess:
    def __init__(self, label_array: list):
        label_array.extend([Constants.PAD_TOKEN])
        label_array.sort()
        self.label_array = label_array
        self.labelToIndex = {item: index for index, item in enumerate(label_array)}
        self.indexToLabel = {index: item for index, item in enumerate(label_array)}

    @staticmethod
    def encode(label, uniqueLabelArray):
        num_classes = len(uniqueLabelArray)
        one_hot = np.zeros(num_classes)
        index = uniqueLabelArray.index(label)
        one_hot[index] = 1
        return one_hot

    def decode(self, oneHotEncoding):
        label_index = np.argmax(oneHotEncoding)
        return self.indexToLabel[label_index]

    def getIndexOfLabel(self, label):
        return self.labelToIndex[label]

    def getLabelOfIndex(self, index):
        return self.indexToLabel[index]

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
