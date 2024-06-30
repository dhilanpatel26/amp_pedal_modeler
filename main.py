from pathlib import Path
from src.data_processing import DataProcessor
from src.data_manager import DataManager
import numpy as np
import os

CURRENT_FILE = Path(__file__).resolve()
ROOT_DIR = CURRENT_FILE.parent
SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"

class Driver:
    def __init__(self, latency=10):
        self.latency = latency  # milliseconds, used for AudioSegment slicing
        self.overlap = self.latency // 2  # milliseconds, reducing perceived latency by 50%
        self.actual_sample_window = int(self.latency / 1000 * 48000)  # samples, used for torch.nn.Conv1d
        self.perceived_sample_window = int((self.latency - self.overlap) / 1000 * 48000)  # samples, used for torch.nn.Conv1d
        print(f"Actual sample window: {self.actual_sample_window}")
        print(f"Perceived sample window: {self.perceived_sample_window}")

    def main(self):
        # Temporary usage -- will be moved to main.py
        clean_dir = str(DATA_DIR / "raw/clean_guitar_signals")
        fx_dir = str(DATA_DIR / "raw/fx_guitar_signals")
        segmented_clean_dir = str(DATA_DIR / "processed/segmented/segmented_clean")
        segmented_fx_dir = str(DATA_DIR / "processed/segmented/segmented_fx")

        DataProcessor.create_and_save_segmented_wavs(clean_dir, fx_dir, segmented_clean_dir, segmented_fx_dir, self.latency, self.overlap)

        seg_save_path = str(DATA_DIR / "processed/segmented/segmented.npz")  # this is the file name without extension?
        DataProcessor.create_and_save_waveforms_dict(segmented_clean_dir, segmented_fx_dir, seg_save_path)

        # # testing only
        # test_waveform = DataProcessor.load_waveform_pairs(waveforms_path + ".npz")
        # temp_clean_file = str(DATA_DIR / "processed/temp_clean_file")
        # temp_amplified_file = str(DATA_DIR / "processed/temp_amplified_file")
        # np.savetxt(temp_clean_file, test_waveform[list(test_waveform.keys())[0]]["clean"], delimiter=',')
        # np.savetxt(temp_amplified_file, test_waveform[list(test_waveform.keys())[0]]["amplified"], delimiter=',')

        train_dir: Path = DATA_DIR / "processed/train"
        val_dir: Path = DATA_DIR / "processed/val"
        test_dir: Path = DATA_DIR / "processed/test"

        DataManager.split_and_save_data(train_dir, val_dir, test_dir)

if __name__ == "__main__":
    driver = Driver(latency=10)
    driver.main()