import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, default='../physionet.org/files/voiced/1.0.0' ,help='File path to text data')
    parser.add_argument('--cleanedPath', type=str, default='patient_clinical_notes', help='File path to store cleaned data')
    args = parser.parse_args()

    # Create cleanedPath directory
    cleanedPath = os.path.join("..", args.cleanedPath)
    os.makedirs(cleanedPath, exist_ok=True)

    # Select patient clinical notes (ending in -info.txt) and remove line with Diagnosis
    for filename in os.listdir(args.dataPath):
        if filename.endswith('-info.txt'):
            data_path = os.path.join(args.dataPath, filename)
            cleaned_path = os.path.join(cleanedPath, filename)

            with open(data_path, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
                filtered_lines = [line for line in lines if "Diagnosis:" not in line]
    
            with open(cleaned_path, 'w', encoding='utf-8') as outfile:
                outfile.writelines(filtered_lines)