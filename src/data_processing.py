from pydub import AudioSegment
import os
import torchaudio
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from src.src_paths import DATA_DIR

class DataProcessor:

    @staticmethod
    def segment_audio(input_path, output_dir, segment_length, overlap):
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
        # AudioSegment indices are in milliseconds
        audio: AudioSegment = AudioSegment.from_file(input_path)

        num_samples = len(audio)
        step_size = segment_length - overlap

        for i in range(0, num_samples - segment_length + 1, step_size):
            segment = audio[i:i + segment_length]

            # Calculate segment start and end time in milliseconds
            start_time = i
            end_time = i + segment_length

            segment_path = os.path.join(output_dir, f'segment_{start_time}_{end_time}.wav')

            # Export segment using pydub
            segment.export(segment_path, format='wav')

    @staticmethod
    def generate_waveform_dict(segmented_clean_dir, segmented_fx_dir):  # segment_audio must first be called to generate .wavs
        waveforms: Dict[str, Dict[str, np.ndarray]] = {}
        for clean_file, amplified_file in zip(os.listdir(segmented_clean_dir), os.listdir(segmented_fx_dir)):
            clean_file_path = os.path.join(segmented_clean_dir, clean_file)
            amplified_file_path = os.path.join(segmented_fx_dir, amplified_file)
            clean_waveform, _ = DataProcessor.get_numpy_waveform(AudioSegment.from_file(clean_file_path))
            amplified_waveform, _ = DataProcessor.get_numpy_waveform(AudioSegment.from_file(amplified_file_path))
            waveforms[clean_file] = {
                "clean": clean_waveform,
                "amplified": amplified_waveform
            }
        return waveforms

    @staticmethod
    def get_numpy_waveform(audio_segment: AudioSegment) -> Tuple[np.ndarray, int]:  # TODO: check typing here
        if audio_segment.channels == 2:  # stereo sound
            audio_segment.set_channels(1)  # convert to mono
        waveform = np.array(audio_segment.get_array_of_samples())
        waveform = DataProcessor.standard_normalization(waveform)  # normalizes arrays to [-1, 1]
        return waveform, audio_segment.frame_rate  # sampling rate should be 48000 Hz
        
    @staticmethod    
    def standard_normalization(waveform: np.ndarray) -> np.ndarray:
        # smart normalization to [-1, 1]
        segment_max = np.max(np.abs(waveform))
        if segment_max:
            return waveform / segment_max
        else: return waveform
        
    @staticmethod
    def save_waveform_pairs(waveform_pairs, save_path) -> None:
        np.savez_compressed(save_path, **waveform_pairs)

    @staticmethod
    def load_waveform_pairs(load_path) -> Dict[str, np.ndarray]:
        data = np.load(load_path, allow_pickle=True)
        return {key: data[key].item() for key in data}

    @staticmethod
    def create_and_save_segmented_wavs(clean_dir, fx_dir, segmented_clean_dir, segmented_fx_dir, segment_length, overlap):
        # segment all files in the directory
        for filename in os.listdir(clean_dir):
            if filename.endswith('.aif'):
                input_path = os.path.join(clean_dir, filename)
                DataProcessor.segment_audio(input_path, segmented_clean_dir, segment_length, overlap)

        for filename in os.listdir(fx_dir):
            if filename.endswith('.aif'):
                input_path = os.path.join(fx_dir, filename)
                DataProcessor.segment_audio(input_path, segmented_fx_dir, segment_length, overlap)

    def create_and_save_waveforms_dict(segmented_clean_dir, segmented_fx_dir, save_path):
        waveforms = DataProcessor.generate_waveform_dict(segmented_clean_dir, segmented_fx_dir)
        DataProcessor.save_waveform_pairs(waveforms, save_path)
