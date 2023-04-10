from word_to_vector import WordToVector

class DnnInference:
    def __init__(self, testWordArray, testLabelArray):
        self.testWordArray = testWordArray
        self.testLabelArray = testLabelArray

    def inferenceAndPrint(self, model, limit: int):
        for i in range(limit):
            currentWord = self.testWordArray[i]
            wordToVec = WordToVector.getPretrainedWordToVec(currentWord)
            predictions = model.predict(wordToVec)
            print(currentWord, self.testLabelArray[i], predictions)

    @staticmethod
    def inferenceFromTrainedWordToVec(wordToVecList, model, limit: int):
        for i in range(limit):
            predictions = model.predict(wordToVecList[i])
            print(predictions)

