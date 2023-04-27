from src.preprocess import PreProcess


class LabelProcess:
    def __init__(self, labelFileName):
        unique_labels, id2label, label2id = PreProcess.readLabelInfo(
            labelFileName, True
        )
        self.unique_labels = unique_labels
        self.id2label = id2label
        self.label2id = label2id

    def getId2label(self, id):
        return self.id2label[id]

    def getLabel2id(self, label):
        return self.label2id[label]
