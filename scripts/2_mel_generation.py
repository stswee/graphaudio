import os
import argparse
import numpy as np
import wfdb
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_recording(base_path):
    record = wfdb.rdrecord(base_path, channels=[0])
    return record.p_signal.flatten()

def segment_signal(signal, remove_samples, segment_length, overlap):
    signal = signal[remove_samples:]  # Remove initial silence
    segments = []
    start = 0
    while start + segment_length <= len(signal):
        segment = signal[start:start + segment_length]
        segments.append(segment)
        start += (segment_length - overlap)
    return segments

def extract_mel_spectrogram(signal_segment, fs, n_fft=1024, hop_length=256, n_mels=64):
    # Compute Mel spectrogram (power)
    mel_spec = librosa.feature.melspectrogram(y=signal_segment, sr=fs, n_fft=n_fft,
                                              hop_length=hop_length, n_mels=n_mels, power=2.0)
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

def preprocess_all_records(input_dir, output_dir, fs, remove_sec, segment_length, overlap):
    os.makedirs(output_dir, exist_ok=True)
    remove_samples = int(fs * remove_sec)

    for file in tqdm(os.listdir(input_dir), desc="Processing records"):
        if file.endswith(".hea"):
            record_name = os.path.splitext(file)[0]
            full_path = os.path.join(input_dir, record_name)

            try:
                signal = load_recording(full_path)
                segments = segment_signal(signal, remove_samples, segment_length, overlap)

                mel_specs = []
                for segment in segments:
                    mel_spec = extract_mel_spectrogram(segment, fs)
                    mel_specs.append(mel_spec)

                # Stack mel spectrograms horizontally (along time axis)
                # Each mel_spec shape: (n_mels, time_frames)
                stacked_mel = np.hstack(mel_specs)  # Shape: (n_mels, total_time_frames)

                # Save stacked mel spectrogram as npy
                out_filename = f"{record_name}_stacked_mel.npy"
                out_path = os.path.join(output_dir, out_filename)
                np.save(out_path, stacked_mel)

                print(f"Processed {record_name}: stacked Mel spectrogram shape {stacked_mel.shape}")

            except Exception as e:
                print(f"Error processing {record_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract stacked Mel spectrograms from VOICED dataset.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with .hea/.dat recordings.")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination folder for stacked Mel spectrograms.")
    parser.add_argument("--fs", type=int, default=8000, help="Sampling frequency (default: 8000 Hz).")
    parser.add_argument("--remove_sec", type=float, default=0.15, help="Seconds to remove from start (default: 0.15s).")
    parser.add_argument("--segment_length", type=int, default=8192, help="Length of each segment in samples.")
    parser.add_argument("--overlap", type=int, default=4096, help="Overlap between segments in samples.")
    parser.add_argument("--n_fft", type=int, default=1024, help="FFT window size for Mel spectrogram.")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop length for Mel spectrogram.")
    parser.add_argument("--n_mels", type=int, default=64, help="Number of Mel bands.")

    args = parser.parse_args()

    preprocess_all_records(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fs=args.fs,
        remove_sec=args.remove_sec,
        segment_length=args.segment_length,
        overlap=args.overlap
    )

if __name__ == "__main__":
    main()
