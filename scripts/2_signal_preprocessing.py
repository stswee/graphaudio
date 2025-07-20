import os
import argparse
import numpy as np
import wfdb

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

def preprocess_all_records(input_dir, output_dir, fs, remove_sec, segment_length, overlap):
    os.makedirs(output_dir, exist_ok=True)
    remove_samples = int(fs * remove_sec)

    for file in os.listdir(input_dir):
        if file.endswith(".hea"):
            record_name = os.path.splitext(file)[0]
            full_path = os.path.join(input_dir, record_name)

            try:
                signal = load_recording(full_path)
                segments = segment_signal(signal, remove_samples, segment_length, overlap)

                for i, segment in enumerate(segments):
                    out_filename = f"{record_name}_seg{i}.npy"
                    out_path = os.path.join(output_dir, out_filename)
                    np.save(out_path, segment)

                print(f"Processed {record_name}: {len(segments)} segments saved.")

            except Exception as e:
                print(f"Error processing {record_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess VOICED dataset audio recordings.")
    parser.add_argument("--input_dir", type=str, required=True, help="Folder with .hea/.dat recordings.")
    parser.add_argument("--output_dir", type=str, required=True, help="Destination folder for segments.")
    parser.add_argument("--fs", type=int, default=8000, help="Sampling frequency (default: 8000 Hz).")
    parser.add_argument("--remove_sec", type=float, default=0.15, help="Seconds to remove from start (default: 0.15s).")
    parser.add_argument("--segment_length", type=int, default=8192, help="Length of each segment in samples.")
    parser.add_argument("--overlap", type=int, default=4096, help="Overlap between segments in samples.")

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
