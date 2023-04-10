import spacy as spacy
from gensim.models import Word2Vec
import numpy as np

class WordToVector:
    def __init__(self, sentences: list[list[str]]):
        self.sentences = sentences
        self.zero_vector = np.zeros(100)
        self.model: Word2Vec = Word2Vec(sentences, min_count=1, vector_size=100, workers=4)

    def getTrainedWordToVec(self, word: str):
        # Get the vector for a word
        try:
            vector = self.model.wv[word]
            return vector
        except KeyError:
            vector = self.zero_vector
            return vector

    @staticmethod
    def getPretrainedWordToVec(word: str):
        # Load a pre-trained model
        nlp = spacy.load("en_core_web_md")
        vector = nlp(word).vector
        # print(vector)
        return vector

    @staticmethod
    def getPretrainedWordToVecList(wordArr: list[str]):
        wordToVecArr = []
        nlp = spacy.load("en_core_web_md")
        for word in wordArr:
            wordToVecArr.append(nlp(word).vector)
        return wordToVecArr
