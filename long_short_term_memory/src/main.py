import torch
from torch.utils.data import TensorDataset, DataLoader
import lightning as L
from lstm_from_lightening import LightningLSTM

def getDataLoader():
    inputs = torch.tensor([[0., 0.5, 0.25, 1., 0.5], [1., 0.5, 0.25, 1.]])
    labels = torch.tensor([0., 1.])

    dataset = TensorDataset(inputs, labels)
    # num_workers = 10
    dataloader = DataLoader(dataset)
    return dataloader

def train():
    model = LightningLSTM()
    check(model)
    trainer = L.Trainer(max_epochs=300, log_every_n_steps=2)

    trainer.fit(model, train_dataloaders=getDataLoader())

    print("After optimization, the parameters are...")
    for name, param in model.named_parameters():
        print(name, param.data)

    print("\nNow let's compare the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())


def check(model):
    print("Before optimization, the parameters are...")
    for name, param in model.named_parameters():
        print(name, param.data)

    print("\nNow let's compare the observed and predicted values...")
    print("Company A: Observed = 0, Predicted =", model(torch.tensor([0., 0.5, 0.25, 1.])).detach())
    print("Company B: Observed = 1, Predicted =", model(torch.tensor([1., 0.5, 0.25, 1.])).detach())
    return model


if __name__ == "__main__":
    train()
