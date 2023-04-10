from util.constants import Constants


class OneHotEncoder:
    def __init__(self):
        pass

    @staticmethod
    def getOneHotEncodingOfOutput(label_array):
        oneHotLabel_array = []
        for item in label_array:
            if item == Constants.BEGEN:
                oneHotLabel_array.append(Constants.BEGEN_ONEHOT)
            if item == Constants.INSIDE:
                oneHotLabel_array.append(Constants.INSIDE_ONEHOT)
            if item == Constants.OUTSIDE:
                oneHotLabel_array.append(Constants.OUTSIDE_ONEHOT)
        return oneHotLabel_array
