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

    def train_one_epoch(self, training_pairs):
        self.model.train()  # set model to training mode
        for key in training_pairs:
            clean_tensor = torch.tensor(training_pairs[key]['clean']).unsqueeze(0).unsqueeze(0).to(self.device)
            fx_tensor = torch.tensor(training_pairs[key]['amplified']).unsqueeze(0).unsqueeze(0).to(self.device)
            # print(f"{len(waveform_pairs[key]['amplified'])}")
            # print(f"clean_tensor shape: {clean_tensor.shape}")
            # print(f"fx_tensor shape: {fx_tensor.shape}")

            # type conversion for consistency among tensors
            clean_tensor = clean_tensor.float()
            fx_tensor = fx_tensor.float()

            # forward pass
            outputs = self.model(clean_tensor)
            loss = self.criterion(outputs, fx_tensor)

            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()
            
    def train_epochs(self, epochs, training_pairs, validation_pairs, patience=3):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(training_pairs)
            val_loss = self.validate(validation_pairs)

            print(f"Epoch {epoch + 1} completed, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Check if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # reset patience counter
                # TODO: Save model checkpoint here
            else:
                patience_counter += 1  # increment patience counter

            # Early stopping check
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch + 1}")
                break

    def validate(self, validation_pairs):
        self.model.eval()  # set model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():
            for key in validation_pairs:
                clean_tensor = torch.tensor(validation_pairs[key]['clean']).unsqueeze(0).unsqueeze(0).to(self.device)
                fx_tensor = torch.tensor(validation_pairs[key]['amplified']).unsqueeze(0).unsqueeze(0).to(self.device)
                clean_tensor = clean_tensor.float()
                fx_tensor = fx_tensor.float()
                outputs = self.model(clean_tensor)
                loss = self.criterion(outputs, fx_tensor)
                total_val_loss += loss.item()
        return total_val_loss / len(validation_pairs)