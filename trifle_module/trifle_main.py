#!/usr/bin/env python

# MAIN SCRIPT FOR RUNNING TRIFLE 
# Version 11-06-2025 by TJ de Kloe 

import argparse
import sys
import os
from pathlib import Path
import pickle
import logging
from datetime import datetime
import zipfile

import trifle_module.trifle_spatial as trifle_spatial
import trifle_module.trifle_temporal as trifle_temporal
import trifle_module.trifle_timevaryingmixing as trifle_tvM

def setup_logging(output_dir):
    log_path                = output_dir / "trifle_run.log"

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            filename        =log_path,
            level           =logging.INFO,
            format          ="%(asctime)s - %(levelname)s - %(message)s"
        )

    console                 = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter               = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

def zip_output(output_dir):
    zip_path                = output_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder, _, files in os.walk(output_dir):
            for file in files:
                full_path   = os.path.join(folder, file)
                relative_path = os.path.relpath(full_path, output_dir)
                zipf.write(full_path, arcname=relative_path)
    logging.info(f"\nZipped output directory to: {zip_path}")

def main():
    parser                  = argparse.ArgumentParser(description="Run full TRIFLE pipeline (group-level only).")
    parser.add_argument("--input", required=True, help="Text file listing preprocessed .nii.gz files for group ICA.")
    parser.add_argument("--tr", required=True, type=float, help="Repetition time in seconds.")
    parser.add_argument("--output", required=True, help="Directory for outputs.")
    parser.add_argument("--spatial_modelorder", type=int, default=20, help="Model order for spatial ICA.")
    parser.add_argument("--temporal_modelorder", type=int, required=True, help="Model order for temporal ICA.")
    parser.add_argument("--mask", type=str, required=True, help="Path to brain mask.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for temporal ICA reproducibility.")

    args                    = parser.parse_args()
    output_dir              = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(output_dir)
    logging.info(f"===== Starting TRIFLE run: {datetime.now().isoformat()} =====")

    try:
        # Step 1: Spatial ICA
        spatial_output_dir  = output_dir / "spatial"
        spatial_output_dir.mkdir(exist_ok=True)
        trifle_spatial.run_spatial_ica(
            input_files     =args.input,
            output_dir      =str(spatial_output_dir),
            tr              =args.tr,
            n_components    =args.spatial_modelorder,
            bet             =False,
            mask            =args.mask
        )
        S_path = spatial_output_dir / "melodic_IC.nii.gz"
        T_path = spatial_output_dir / "melodic_mix"

        # Step 2: Temporal ICA
        temporal_output_dir = output_dir / "temporal"
        temporal_output_dir.mkdir(exist_ok=True)
        trifle_temporal.run_temporal_ica(
            input_path      =T_path,
            output_path     =temporal_output_dir,
            model_order     =args.temporal_modelorder,
            seed            =args.seed
        )
        M_path = temporal_output_dir / f"tica_k{args.temporal_modelorder}" / "M.pkl"
        B_path = temporal_output_dir / f"tica_k{args.temporal_modelorder}" / "B.pkl"

        # Step 3: TRIFLE Core Analysis
        final_dir           = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        trifle_tvM.run_trifle_analysis(
            input_file      =args.input,
            S_file          =S_path,
            T_file          =T_path,
            M_file          =M_path,
            B_file          =B_path,
            mask_file       =args.mask,
            output_dir      =final_dir
        )

        logging.info("\nFinal results saved in 'final' subfolder.")
        zip_output(output_dir)

    except Exception as e:
        logging.exception(f"\nTRIFLE run failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()