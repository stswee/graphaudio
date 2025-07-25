import os
import argparse
import numpy as np
from collections import defaultdict

def stack_mels(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    # List all mel npy files
    files = [f for f in os.listdir(input_folder) if f.endswith('.npy')]

    # Group files by patient prefix (e.g., voice001)
    patient_files = defaultdict(list)
    for f in files:
        # Assuming filename format: voice001_seg1_mel.npy
        patient_id = f.split('_')[0]  # 'voice001'
        patient_files[patient_id].append(f)

    for patient_id, seg_files in patient_files.items():
        # Sort segments to maintain temporal order
        seg_files = sorted(seg_files)

        mel_segments = []
        for seg_file in seg_files:
            mel_path = os.path.join(input_folder, seg_file)
            mel = np.load(mel_path)  # shape (64, 33)
            mel_segments.append(mel)

        # Stack along time axis (axis=1)
        stacked_mel = np.concatenate(mel_segments, axis=1)  # shape (64, 33 * num_segments)

        # Save stacked mel spectrogram
        out_path = os.path.join(output_folder, f"{patient_id}_stacked_mel.npy")
        np.save(out_path, stacked_mel)

        print(f"Saved stacked mel spectrogram for {patient_id} to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Stack mel spectrogram segments per patient.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing mel spectrogram .npy files")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save stacked mel spectrograms")
    args = parser.parse_args()

    stack_mels(args.input_folder, args.output_folder)

if __name__ == "__main__":
    main()
