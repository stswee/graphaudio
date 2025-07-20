import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from panns_inference import AudioTagging

def generate_embeddings(input_folder, output_folder, device):
    os.makedirs(output_folder, exist_ok=True)

    # Load pretrained CNN14 model wrapper
    model = AudioTagging(checkpoint_path=None, device=device)

    # Get all mel files
    mel_files = [f for f in os.listdir(input_folder) if f.endswith('_mel.npy')]

    # Progress bar
    for fname in tqdm(mel_files, desc="Generating embeddings", unit="file"):
        mel_path = os.path.join(input_folder, fname)
        mel = np.load(mel_path)

        # Expected shape: (1, mel_bins, time_frames)
        input_tensor = torch.from_numpy(mel).float().unsqueeze(0).to(device)

        with torch.no_grad():
            output_dict = model.forward(input_tensor)
            embedding = output_dict['embedding']

        # Save embedding
        out_name = fname.replace('_mel.npy', '_embed.npy')
        out_path = os.path.join(output_folder, out_name)
        np.save(out_path, embedding.cpu().numpy())

    print(f"\nSaved {len(mel_files)} embeddings to: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="Convert Mel spectrograms to audio embeddings using PANNs (CNN14).")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing mel spectrogram .npy files")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save audio embeddings")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference")

    args = parser.parse_args()
    generate_embeddings(args.input_folder, args.output_folder, args.device)

if __name__ == "__main__":
    main()
