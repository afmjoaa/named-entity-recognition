from model import DnnModel
from keras.models import load_model
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping


class DnnTraining:
    def __init__(self, input_dim=300, output_dim=3):
        self.model = DnnModel().createDefaultModel(input_dim, output_dim)

    @staticmethod
    def f1_score(y_true, y_pred):
        """
        Custom F1 score metric function.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1

    def startTraining(self, X_train, y_train, X_val, y_val, epochs=10):
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', DnnTraining.f1_score])

        # Create a TensorBoard callback
        log_dir = "../logs"
        tensorboard_callback = TensorBoard(log_dir=log_dir)
        # Define the early stopping callback
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), callbacks=[tensorboard_callback, early_stopping_callback])

    def saveTrainedModel(self):
        self.model.save('dnn_model.h5')

    def getCurrentModel(self):
        return self.model

    @staticmethod
    def getSavedModel():
        dnnModel = load_model('dnn_model.h5')
        return dnnModel



