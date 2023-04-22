import spacy
from utility import Constants
from spacy.util import filter_spans


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
    def formatMapForNer(results):
        sentence = ""
        entities = []
        start = 0
        for word, label in results.items():
            if label != "O":
                entities.append((start, len(sentence) + len(word), label))
            sentence += word + " "
            start = len(sentence)
        return {"sentence": sentence.strip(), "entities": entities}

    @staticmethod
    def createDocFromFormattedMap(formattedMap, nlp: spacy.language):
        text = formattedMap["sentence"]
        labels = formattedMap["entities"]
        doc = nlp.make_doc(text)
        entities = []
        for start, end, label in labels:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                entities.append(span)
        filtered_entities = filter_spans(entities)
        doc.ents = filtered_entities
        return doc
