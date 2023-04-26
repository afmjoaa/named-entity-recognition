class Constants:
    SEPERATOR = " _ _ "
    ID_IDENTIFIER = "#"
    BIOX_SEPERATOR = "-"
    BEGEN = "B"
    INSIDE = "I"
    OUTSIDE = "O"
    BEGEN_ONEHOT = [1, 0, 0]
    INSIDE_ONEHOT = [0, 1, 0]
    OUTSIDE_ONEHOT = [0, 0, 1]
    PAD_TOKEN = '<PAD>'
    START_TOKEN = '<START>'
    END_TOKEN = '<END>'
    SEQUENCE_LENGTH = 35
    TASK = "ner"
    MODEL_CHECKPOINT = "distilbert-base-uncased"
    BATCH_SIZE = 16


class Utility:
    @staticmethod
    def split_label_text(text) -> tuple[str, str]:
        parts = text.split("-")
        return parts[0], "-" + parts[1]

    @staticmethod
    def make_sentence(words) -> str:
        sentence = " ".join(words)
        return sentence
