from src.utils import Constants


class PreProcess:
    @staticmethod
    def getDataArrayAsMap(dataFile: str, onlyBioTagging=False):
        # Open the CoNLL file
        data_array = []
        with open(dataFile, encoding='utf-8') as f:
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
    def formatMapForNer(results):
        sentence = ""
        tag = ""
        for word, label in results.items():
            sentence += word + " "
            tag += label + " "

        return {"sentence": sentence.strip(), "tag": tag.strip()}