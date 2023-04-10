from pre_process import PreProcess
from word_to_vector import WordToVector

def inference():
    # Load the trained model and inference the dev data

    # Start the training save the model to a file.
    wordArr, labelArr = PreProcess.getTrainingTuple(dataFile='dummy.conll', onlyBioTagging=True)
    print(wordArr, labelArr)

    # wordToVector = WordToVector(sentenceArray).getTrainedWordToVec("winner")

    # wordToVector = WordToVector.getPretrainedWordToVec("this")
    # print(sentenceArray)
    # print(wordToVector)


if __name__ == "__main__":
    inference()
