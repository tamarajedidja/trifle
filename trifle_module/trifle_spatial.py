#!/usr/bin/env python

# TRIFLE - SPATIAL DECOMPOSITION MODULE
# Version: 11-06-2025 by TJ de Kloe 

import argparse
import logging
import subprocess
from pathlib import Path

# ---------- Set up module-level logger ----------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True

def run_spatial_ica(input_list_file, output_dir, tr, n_components=20, bet=False, mask=None):
    input_path = Path(input_list_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file list not found: {input_path}")
    
    with open(input_path) as f:
        file_lines = [line.strip() for line in f if line.strip()]
        logger.info("Files to be included in group ICA:")
        for path in file_lines:
            logger.info(f"  {path}")

    logger.info("Running group spatial ICA with parameters:")
    logger.info(f"  Input list: {input_list_file}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  TR: {tr}")
    logger.info(f"  Components: {n_components}")
    logger.info(f"  BET: {bet}")
    logger.info(f"  Mask: {mask}")

    cmd = [
        "melodic",
        "-i", input_list_file,
        "-o", output_dir,
        f"--tr={tr}",
        "-a", "concat",
        "--sep_vn",
        "--disableMigp",
        "--report",
        "--Oall",
        "-d", str(n_components)
    ]
    if not bet:
        cmd.append("--nobet")
    if mask:
        cmd.extend(["-m", mask])

    logger.info("Executing MELODIC command:")
    logger.info("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run group spatial ICA using TRIFLE.")
    parser.add_argument("--input", required=True, help="Text file listing paths to 4D preprocessed functional data (.nii.gz).")
    parser.add_argument("--output", required=True, help="Directory for outputs.")
    parser.add_argument("--tr", required=True, type=float, help="Repetition time in seconds.")
    parser.add_argument("--n_components", type=int, default=20, help="Number of ICA components.")
    parser.add_argument("--bet", action="store_true", help="Enable BET brain extraction during ICA.")
    parser.add_argument("--mask", type=str, default=None, help="Optional brain mask for MELODIC.")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logfile_path = output_dir / "trifle_spatial.log"
    logging.basicConfig(filename=logfile_path, filemode='w', level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("===== Starting trifle_spatial.py =====")
    run_spatial_ica(args.input, args.output, args.tr, args.n_components, args.bet, args.mask)
    logger.info("===== trifle_spatial.py finished successfully =====")

if __name__ == "__main__":
    main()