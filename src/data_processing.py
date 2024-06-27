from pydub import AudioSegment
import os
import torchaudio
import torch
import numpy as np
from pathlib import Path
from typing import Dict

CURRENT_FILE = Path(__file__).resolve()
SRC_DIR = CURRENT_FILE.parent
ROOT_DIR = SRC_DIR.parent
DATA_DIR = ROOT_DIR / "data"

def segment_audio(input_path, output_dir, segment_length=1024, overlap=0):
    """
    Segments an audio file into smaller chunks and saves them using pydub.
    
    Args:
        input_path (str): Path to the input audio file.
        output_dir (str): Directory to save the segments.
        segment_length (int): Length of each segment in milliseconds.
        overlap (int): Number of overlapping milliseconds between segments.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load AIFF file using pydub
    audio: AudioSegment = AudioSegment.from_file(input_path)

    num_samples = len(audio)
    step_size = segment_length - overlap

    waveforms: Dict[np.array]

    for i in range(0, num_samples - segment_length + 1, step_size):
        segment = audio[i:i + segment_length]

        # Calculate segment start and end time in milliseconds
        start_time = i
        end_time = i + segment_length

        segment_path = os.path.join(output_dir, f'segment_{start_time}_{end_time}.wav')

        # Export segment using pydub
        segment.export(segment_path, format='wav')

def get_numpy_waveform(audio_segment: AudioSegment):
    if audio_segment.channels == 2:  # stereo sound
        audio_segment.set_channels(1)  # convert to mono
    waveform = np.array(audio_segment.get_array_of_samples())
    waveform = standard_normalization(waveform)
    return waveform, audio_segment.frame_rate  # sampling rate should be 48000 Hz
    
def standard_normalization(waveform):
    return waveform / np.max(np.abs(waveform))


# Temporary usage -- will be moved to main.py
clean_dir = str(DATA_DIR / "raw/clean_guitar_signals")
fx_dir = str(DATA_DIR / "raw/fx_guitar_signals")
segmented_clean_dir = str(DATA_DIR / "processed/segmented/segmented_clean")
segmented_fx_dir = str(DATA_DIR / "processed/segmented/segmented_fx")

# Segment all files in the directory
for filename in os.listdir(clean_dir):
    if filename.endswith('.aif'):
        input_path = os.path.join(clean_dir, filename)
        segment_audio(input_path, segmented_clean_dir)

for filename in os.listdir(fx_dir):
    if filename.endswith('.aif'):
        input_path = os.path.join(fx_dir, filename)
        segment_audio(input_path, segmented_fx_dir)
