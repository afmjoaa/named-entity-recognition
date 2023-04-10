from src.util.constants import Constants


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


