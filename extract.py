#!/usr/bin/env python3
"""
extract_edf_pairs.py

Find up to N EDF PSG files and their hypnogram matches in a dataset folder,
copy them to an output folder, and optionally create a compressed archive.

Usage example:
    python3 extract_edf_pairs.py --dataset /path/to/dataset --out /path/to/output --n 20 --zip
"""

import argparse
import os
import re
import random
import shutil
from pathlib import Path

def find_files(dataset_dir):
    p = Path(dataset_dir)
    all_files = [f for f in p.iterdir() if f.is_file()]
    return all_files

def is_edf(file: Path):
    return file.suffix.lower() == '.edf'

def looks_like_hyp(file: Path):
    name = file.stem.lower()
    return 'hypnogram' in name or 'hyp' in name or 'stage' in name

def candidate_base_names(name: str):
    """
    Create candidate base keys from a filename stem to help matching.
    Example: "SC4321E0-PSG" -> ["sc4321e0", "sc4321e0 psg", "sc4321e0"]
    """
    name = name.lower()
    # remove common suffix tokens
    tokens = re.split(r'[-_.\s]+', name)
    if tokens:
        return [''.join(tokens), tokens[0]] + tokens
    return [name]

def match_hyp_for_psg(psg_file: Path, all_files):
    """
    Heuristic search for a hyp file matching the given PSG file.
    Returns Path or None.
    """
    psg_stem = psg_file.stem
    bases = candidate_base_names(psg_stem)

    # priority 1: file whose stem contains base and 'hyp' or 'hypnogram' or 'stage'
    for bf in bases:
        for f in all_files:
            if f == psg_file:
                continue
            stem = f.stem.lower()
            if bf in stem and looks_like_hyp(f):
                return f

    # priority 2: any file that contains 'hypnogram' and shares the same numeric ID
    digits = re.findall(r'\d+', psg_stem)
    if digits:
        for d in digits:
            for f in all_files:
                if f == psg_file:
                    continue
                if d in f.stem and looks_like_hyp(f):
                    return f

    # priority 3: any file with 'hyp' in it (fallback)
    for f in all_files:
        if f == psg_file:
            continue
        if looks_like_hyp(f):
            return f

    return None

def gather_pairs(dataset_dir, n=20):
    all_files = find_files(dataset_dir)
    edf_psg_files = [f for f in all_files if is_edf(f) and 'hyp' not in f.stem.lower() and 'hypnogram' not in f.stem.lower()]
    pairs = []
    used_hyps = set()

    # Shuffle for random sampling
    random.shuffle(edf_psg_files)

    for psg in edf_psg_files:
        hyp = match_hyp_for_psg(psg, all_files)
        if hyp and hyp not in used_hyps:
            pairs.append((psg, hyp))
            used_hyps.add(hyp)
        if len(pairs) >= n:
            break

    return pairs

def copy_pairs(pairs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for psg, hyp in pairs:
        # keep original filenames
        dst_psg = out_dir / psg.name
        dst_hyp = out_dir / hyp.name
        shutil.copy2(psg, dst_psg)
        shutil.copy2(hyp, dst_hyp)
        copied.append((dst_psg, dst_hyp))
    return copied

def make_archive(out_dir, archive_path, fmt='zip'):
    base_name = str(Path(archive_path).with_suffix(''))
    root_dir = str(Path(out_dir).parent)
    base_dir = str(Path(out_dir).name)
    shutil.make_archive(base_name, fmt, root_dir=root_dir, base_dir=base_dir)
    return f"{base_name}.{fmt}"

def main():
    parser = argparse.ArgumentParser(description="Extract EDF PSG + hypnogram pairs.")
    parser.add_argument('--dataset', required=True, help="Path to dataset folder")
    parser.add_argument('--out', required=True, help="Output folder where selected files will be copied")
    parser.add_argument('--n', type=int, default=20, help="Number of pairs to extract (default 20)")
    parser.add_argument('--zip', action='store_true', help="Create a zip archive of the output folder")
    parser.add_argument('--tar', action='store_true', help="Create a tar.gz archive instead of zip")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducible selection")
    args = parser.parse_args()

    random.seed(args.seed)

    pairs = gather_pairs(args.dataset, n=args.n)
    if not pairs:
        print("No PSG-hypnogram pairs found with the heuristics. Check file names / patterns.")
        return

    print(f"Found {len(pairs)} pairs. Copying to {args.out} ...")
    copied = copy_pairs(pairs, args.out)
    for p, h in copied:
        print(f"  - {p.name}  +  {h.name}")

    archive_file = None
    if args.zip or args.tar:
        fmt = 'zip' if args.zip else 'gztar'
        archive_path = str(Path(args.out).with_suffix(''))
        print("Creating archive ...")
        # shutil.make_archive uses format 'zip' or 'gztar'
        archive_file = shutil.make_archive(archive_path, fmt, root_dir=str(Path(args.out).parent), base_dir=str(Path(args.out).name))
        print("Archive created:", archive_file)

    print("Done.")

if __name__ == '__main__':
    main()
