import torch
import torch.nn as nn
import torch.optim as optim
from src.data_processing import DataProcessor
from src.src_paths import DATA_DIR
import numpy as np

class Evaluator:
    def __init__(self, model: nn.Module, criterion=None):
        if not criterion:
            criterion = nn.MSELoss()
        self.criterion = criterion
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)

    def evaluate(self, testing_pairs):  # testing data
        self.model.eval()  # set model to evaluation mode
        mse = np.zeros(len(testing_pairs))
        with torch.no_grad():
            for i, key in enumerate(testing_pairs):
                clean_tensor = torch.tensor(testing_pairs[key]['clean']).unsqueeze(0).unsqueeze(0).to(self.device)
                fx_tensor = torch.tensor(testing_pairs[key]['amplified']).unsqueeze(0).unsqueeze(0).to(self.device)

                # type conversion for consistency among tensors
                clean_tensor = clean_tensor.float()
                fx_tensor = fx_tensor.float()

                # forward pass
                outputs = self.model(clean_tensor)
                loss = self.criterion(outputs, fx_tensor)

                mse[i] = loss.item()
        return mse.mean()