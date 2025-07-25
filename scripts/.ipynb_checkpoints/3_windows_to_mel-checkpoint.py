import os
import argparse
import numpy as np
import librosa
from tqdm import tqdm

def convert_to_mel(input_folder, output_folder, sr, n_mels, n_fft, hop_length):
    os.makedirs(output_folder, exist_ok=True)

    files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]
    print(f"Found {len(files)} segment files to process.")

    for fname in tqdm(files, desc="Converting segments"):
        in_path = os.path.join(input_folder, fname)
        segment = np.load(in_path)

        mel_spec = librosa.feature.melspectrogram(
            y=segment,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        out_fname = fname.replace(".npy", "_mel.npy")
        out_path = os.path.join(output_folder, out_fname)
        np.save(out_path, log_mel_spec)

    print(f"Saved mel spectrograms to {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Convert windowed audio to mel spectrograms.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with windowed .npy audio segments.")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save mel spectrogram .npy files.")
    parser.add_argument("--sr", type=int, default=8000, help="Sampling rate of audio segments.")
    parser.add_argument("--n_mels", type=int, default=64, help="Number of mel bins.")
    parser.add_argument("--n_fft", type=int, default=512, help="FFT window size.")
    parser.add_argument("--hop_length", type=int, default=256, help="Hop length for FFT.")

    args = parser.parse_args()

    convert_to_mel(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        sr=args.sr,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length
    )

if __name__ == "__main__":
    main()
