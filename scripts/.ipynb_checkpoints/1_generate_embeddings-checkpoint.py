import os
import sys
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notesPath", type=str, default='../patient', help='File path to preprocessed text data')
    parser.add_argument("--languageModel", type=str, default='BioBERT', help='Language Model')
    args = parser.parse_args()

    # Select model
    if args.languageModel == 'BioBERT':
        model_name = "dmis-lab/biobert-base-cased-v1.1"
    elif args.languageModel == 'BioClinicalBERT':
        model_name = "emilyalsentzer/Bio_ClinicalBERT"
    else:
        print("Please choose from one of the following models: BioBERT, BioClinicalBERT")
        sys.exit(1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create embeddings output folder
    notes_dir = args.notesPath
    embeddings_root = os.path.abspath(os.path.join(notes_dir, '..', 'embeddings'))
    embeddings_dir = os.path.join(embeddings_root, f"{args.languageModel}_embeddings")
    os.makedirs(embeddings_dir, exist_ok=True)
    print(f"Saving embeddings to: {embeddings_dir}")

    # Get all text files
    files = os.listdir(notes_dir)

    # Loop over preprocessed text data
    for filename in tqdm(files, desc="Generating embeddings"):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(notes_dir, filename)

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token

        save_name = filename.replace(".txt", ".pt")
        torch.save(cls_embedding.cpu(), os.path.join(embeddings_dir, save_name))
