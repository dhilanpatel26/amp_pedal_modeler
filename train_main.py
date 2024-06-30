import torch.nn as nn
import torch.optim as optim
from src.train import Trainer
from src.model import GuitarAmpSimulator

if __name__ == "__main__":
    input_length = 480
    model = GuitarAmpSimulator(input_length)
    loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    trainer = Trainer(model, loss, optimizer)
    trainer.train_epochs(10)