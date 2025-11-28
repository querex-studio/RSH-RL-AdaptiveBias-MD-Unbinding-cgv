#!/usr/bin/env python
import os
import glob

# Input list of residues (one per line), like in the original script
RESID_LIST_FILE = "pi-path-unique-resids-nowat-notype.dat"
# Pattern for Pi contact files
PI_FILES_GLOB = "Pi*.txt"
# Output file
OUTPUT_FILE = "pi-path.dat"


def load_residue_patterns(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Residue list file not found: {path}")
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def count_files_with_pattern(pattern: str, files: list[str]) -> int:
    """
    For a given pattern 'p', count in how many Pi*.txt files
    the pattern appears at least once (like grep | awk | uniq | wc -l).
    """
    count = 0
    for fname in files:
        found = False
        try:
            with open(fname, "r") as fh:
                for line in fh:
                    if pattern in line:
                        found = True
                        break
        except OSError:
            continue
        if found:
            count += 1
    return count


def main():
    residues = load_residue_patterns(RESID_LIST_FILE)
    pi_files = glob.glob(PI_FILES_GLOB)

    if not pi_files:
        print(f"No Pi*.txt files found matching {PI_FILES_GLOB}")
        return

    results: list[tuple[str, int]] = []

    for p in residues:
        num = count_files_with_pattern(p, pi_files)
        results.append((p, num))

    # Sort by second column (count) descending, like `sort -rn -k 2`
    results.sort(key=lambda x: x[1], reverse=True)

    with open(OUTPUT_FILE, "w") as out:
        for p, num in results:
            out.write(f"{p} {num}\n")

    print(f"Wrote {OUTPUT_FILE} with {len(results)} entries.")


if __name__ == "__main__":
    main()
