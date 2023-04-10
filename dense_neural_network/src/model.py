from keras.models import Sequential
from keras.layers import Dense, Dropout

class DnnModel:
    def __init__(self):
        self.model = Sequential()

    def createDefaultModel(self, input_dim, output_dim):
        self.model.add(Dense(128, input_dim=input_dim, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(output_dim, activation='softmax'))
        return self.model
