import torch.nn as nn
import torch.optim as optim
from src.train import Trainer
from src.evaluate import Evaluator
from src.model import GuitarAmpSimulator
from src.data_processing import DataProcessor
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

if __name__ == "__main__":
    input_length = 1440
    model = GuitarAmpSimulator(input_length)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    
    trainer = Trainer(model, criterion, optimizer)
    print("Training model...")
    training_pairs = DataProcessor.load_waveform_pairs(DATA_DIR / "processed/train/train.npz")
    validation_pairs = DataProcessor.load_waveform_pairs(DATA_DIR / "processed/val/val.npz")
    trainer.train_epochs(50, training_pairs, validation_pairs)
    print("Finished training.")

    evaluator = Evaluator(model, criterion)
    testing_pairs = DataProcessor.load_waveform_pairs(DATA_DIR / "processed/test/test.npz")
    print("Evaluating model...")
    mse = evaluator.evaluate(testing_pairs)
    print(f"Mean squared error: {mse:.4f}")