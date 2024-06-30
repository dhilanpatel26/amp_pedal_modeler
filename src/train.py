import torch
import torch.nn as nn
import torch.optim as optim
from src.data_processing import DataProcessor
from src.src_paths import DATA_DIR

class Trainer:

    def __init__(self, model: nn.Module, loss=None, optimizer=None):
        self.model = model
        if not loss:
            loss = nn.MSELoss()
        if not optimizer:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        self.loss = loss
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # function to train the model on data in train.npz
    def train(self):
        # load data
        data = DataProcessor.load_waveform_pairs(str(DATA_DIR / "processed/train/train.npz"))
        # set model to training mode
        self.model.train()
        # iterate over data
        for key in data.keys():
            # get clean and amplified waveforms
            clean = data[key]["clean"]
            amplified = data[key]["amplified"]
            # convert to torch tensors
            clean = torch.tensor(clean, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            amplified = torch.tensor(amplified, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            # zero the gradients
            self.optimizer.zero_grad()
            # forward pass

    # function to train the model for set number of epochs
    def train_epochs(self, num_epochs=10):
        for epoch in range(num_epochs):
            self.train()
            print(f"Epoch {epoch + 1} completed")




