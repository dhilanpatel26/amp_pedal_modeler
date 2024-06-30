from sklearn.model_selection import train_test_split
from src.data_processing import DataProcessor
from src.src_paths import DATA_DIR
from pathlib import Path

class DataManager:

    @staticmethod
    def split_and_save_data(train_dir: Path, val_dir: Path, test_dir: Path):
        # load waveform_pairs from compressed numpy file
        # temporarialy cached so filepath can be hardcoded
        waveform_pairs = DataProcessor.load_waveform_pairs(str(DATA_DIR / "processed/segmented/segmented.npz"))
        data = list(waveform_pairs.items())

        # sklearn to split data into train, validation, and test sets
        train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        # convert back to dictionary
        train_waveform_pairs = dict(train_data)
        val_waveform_pairs = dict(val_data)
        test_waveform_pairs = dict(test_data)

        # save the dictionaries to npz files in processed folder
        DataProcessor.save_waveform_pairs(train_waveform_pairs, str(train_dir / "train.npz"))
        DataProcessor.save_waveform_pairs(val_waveform_pairs, str(val_dir / "val.npz"))
        DataProcessor.save_waveform_pairs(test_waveform_pairs, str(test_dir / "test.npz"))
