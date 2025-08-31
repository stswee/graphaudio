import os
import re
import csv
import argparse

def normalize_key(s: str) -> str:
    """Make a safe, readable column name."""
    s = s.strip()
    s = s.replace("’", "'")
    s = s.replace("‘", "'")
    # remove possessive apostrophes
    s = s.replace("'s", "s")
    # remove remaining apostrophes and commas
    s = s.replace("'", "").replace(",", "")
    # collapse whitespace -> underscore
    s = re.sub(r"\s+", "_", s)
    # remove non-alnum/underscore/-
    s = re.sub(r"[^0-9A-Za-z_\-]", "", s)
    return s.lower()

def clean_value(v: str):
    """Standardize values: strip, map NU/N/A to empty, convert numbers with commas."""
    v = v.strip()
    if not v:
        return ""
    lv = v.lower()
    if lv in {"nu", "n/a", "na", "unknown"}:
        return ""
    # convert european decimal commas (e.g., 1,5 -> 1.5)
    if re.match(r"^-?\d+,\d+$", v):
        v = v.replace(",", ".")
    # try numeric conversion (int or float)
    try:
        if "." in v:
            return float(v)
        else:
            return int(v)
    except ValueError:
        return v

def parse_info_file(path: str):
    """
    Parse a single *-info.txt file into a dict of features.
    Uses the most recent header with empty value (e.g., 'Eating habits:')
    as a section prefix for subsequent key:value lines.
    """
    feats = {}
    current_section = None

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if ":" not in line:
                # not a key:value line — ignore
                continue

            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            if val == "":
                # This is a section header (e.g., 'Eating habits:')
                current_section = normalize_key(key)
                continue

            # Build namespaced key if within a section
            full_key = f"{current_section}__{key}" if current_section else key
            norm_key = normalize_key(full_key)
            feats[norm_key] = clean_value(val)

    return feats

def main():
    parser = argparse.ArgumentParser(description="Extract clinical features from *-info.txt files into a CSV.")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder containing voiceXXX-info.txt files")
    parser.add_argument("--output_csv", type=str, default="../patient/clinical_features.csv", help="Output CSV path")
    args = parser.parse_args()

    rows = []
    all_keys = set()

    for fname in sorted(os.listdir(args.input_folder)):
        if not fname.endswith("-info.txt"):
            continue
        patient_id = fname.split("-")[0]  # voice001-info.txt -> voice001
        path = os.path.join(args.input_folder, fname)

        feats = parse_info_file(path)
        feats_row = {"patient_id": patient_id}
        feats_row.update(feats)
        rows.append(feats_row)
        all_keys.update(feats.keys())

    if not rows:
        print("No *-info.txt files found. Nothing to write.")
        return

    # Build CSV header: patient_id first, then sorted feature keys
    fieldnames = ["patient_id"] + sorted(all_keys)

    # Write CSV
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Wrote {len(rows)} patients with {len(all_keys)} features to {args.output_csv}")

if __name__ == "__main__":
    main()
