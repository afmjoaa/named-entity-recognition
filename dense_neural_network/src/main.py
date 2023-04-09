from pre_process import PreProcess
from word_to_vector import WordToVector

def main():
    sentenceArray = PreProcess.getSentenceArray(dataFile='dummy.conll')
    wordToVector = WordToVector(sentenceArray).getTrainedWordToVec("winner")

    # wordToVector = WordToVector.getPretrainedWordToVec("this")
    print(sentenceArray)
    print(wordToVector)


if __name__ == "__main__":
    main()

