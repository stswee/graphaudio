import os
import argparse
import csv

allowed_labels = [
    "reflux laryngitis",
    "hyperkinetic dysphonia",
    "healthy",
    "hypokinetic dysphonia"
]

def clean_label(raw_label):
    raw_label = raw_label.lower()
    for allowed in allowed_labels:
        if allowed in raw_label:
            return allowed
    return "unknown"  # or choose to skip/handle differently

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', type=str, default='../physionet.org/files/voiced/1.0.0', help='File path to text data')
    parser.add_argument('--cleanedPath', type=str, default='../patient', help='File path to store cleaned data')
    parser.add_argument('--labelPath', type=str, default='../patient/patient_labels.csv', help='Path to save patient labels CSV')
    args = parser.parse_args()

    cleanedPath = args.cleanedPath
    os.makedirs(cleanedPath, exist_ok=True)

    labels = []

    for filename in os.listdir(args.dataPath):
        if filename.endswith('-info.txt'):
            data_path = os.path.join(args.dataPath, filename)
            cleaned_path = os.path.join(cleanedPath, filename)

            patient_id = filename.replace('-info.txt', '')

            with open(data_path, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()

                # ✅ Remove the first row
                if lines:
                    lines = lines[1:]

                filtered_lines = []
                for line in lines:
                    if line.startswith("Diagnosis:"):
                        raw_label = line.strip().replace("Diagnosis:", "").strip()
                        label = clean_label(raw_label)
                        labels.append((patient_id, label))
                    else:
                        filtered_lines.append(line)

            with open(cleaned_path, 'w', encoding='utf-8') as outfile:
                outfile.writelines(filtered_lines)

    # Write labels to CSV
    with open(args.labelPath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['patient_id', 'label'])
        writer.writerows(labels)
