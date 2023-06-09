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


class Utility:
    @staticmethod
    def split_label_text(text) -> tuple[str, str]:
        parts = text.split("-")
        return parts[0], "-" + parts[1]

    @staticmethod
    def make_sentence(words) -> str:
        sentence = " ".join(words)
        return sentence
