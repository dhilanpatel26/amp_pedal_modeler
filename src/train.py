import torch
import torch.nn as nn
import torch.optim as optim
from src.data_processing import DataProcessor
from src.src_paths import DATA_DIR

class Trainer:

    def __init__(self, model: nn.Module, criterion=None, optimizer=None):
        if not criterion:
            criterion = nn.MSELoss()
        if not optimizer:
            optimizer = optim.SGD(model.parameters(), lr=0.01)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def train_one_epoch(self, waveform_pairs):
        for key in waveform_pairs:
            clean_tensor = torch.tensor(waveform_pairs[key]['clean']).unsqueeze(0).unsqueeze(0).to(self.device)
            fx_tensor = torch.tensor(waveform_pairs[key]['amplified']).unsqueeze(0).unsqueeze(0).to(self.device)
            # print(f"{len(waveform_pairs[key]['amplified'])}")
            # print(f"clean_tensor shape: {clean_tensor.shape}")
            # print(f"fx_tensor shape: {fx_tensor.shape}")

            # forward pass
            outputs = self.model(clean_tensor.float())
            loss = self.criterion(outputs, fx_tensor)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()
            
    def train_epochs(self, epochs, waveform_pairs):
        self.model.train()  # set model to training mode
        for epoch in range(epochs):
            loss = self.train_one_epoch(waveform_pairs)
            print(f"Epoch {epoch + 1} completed, Loss: {loss:.4f}")