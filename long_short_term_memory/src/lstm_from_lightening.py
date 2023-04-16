import torch.nn as nn
from torch.optim import Adam
import lightning

class LightningLSTM(lightning.LightningModule):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1)

    def forward(self, feature):
        input_trans = feature.view(len(feature), 1)
        lstm_out, temp = self.lstm(input_trans)

        # lstm_out has the short-term memories for all inputs. We make our prediction with the last one
        prediction = lstm_out[-1]
        return prediction

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=0.1)

    def training_step(self, batch, batch_idx):
        # collect input
        input_i, label_i = batch
        # run input through the neural network
        output_i = self.forward(input_i[0])
        loss = (output_i - label_i) ** 2

        self.log("train_loss", loss)

        if label_i == 0:
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss
