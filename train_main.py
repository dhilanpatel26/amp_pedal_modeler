import torch.nn as nn
import torch.optim as optim
from src.train import Trainer
from src.model import GuitarAmpSimulator
from src.data_processing import DataProcessor
from pathlib import Path

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

if __name__ == "__main__":
    input_length = 480
    model = GuitarAmpSimulator(input_length)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(model, criterion, optimizer)

    waveform_pairs = DataProcessor.load_waveform_pairs(DATA_DIR / "processed/train/train.npz")
    print("Training model...")
    trainer.train_epochs(100, waveform_pairs)