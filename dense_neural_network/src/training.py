from model import DnnModel
from keras.models import load_model

class DnnTraining:
    def __init__(self, input_dim=100, output_dim=3):
        self.model = DnnModel().createDefaultModel(input_dim, output_dim)

    def startTraining(self, X_train, y_train, X_val, y_val, epochs=10):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val))

    def saveTrainedModel(self):
        self.model.save('dnn_model.h5')

    def getCurrentModel(self):
        return self.model

    @staticmethod
    def getSavedModel():
        dnnModel = load_model('my_model.h5')
        return dnnModel



