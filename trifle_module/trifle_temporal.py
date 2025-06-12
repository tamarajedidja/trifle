#!/usr/bin/env python

# TRIFLE - TEMPORAL DECOMPOSITION MODULE
# Version: 11-06-2025 by TJ de Kloe (updated with optional seed and robust logging setup)

import argparse
import pickle
import numpy as np
from pathlib import Path
import logging
from trifle_module import trifle_analysis

# ---------- Logger setup ----------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True  # Let parent logger handle propagation if needed

def run_temporal_ica(input_path, output_path, model_order, seed=42):
    """
    Run temporal ICA on spatial ICA time series (melodic_mix or DR stage 1).
    Saves M, B, and full model for the specified model order.
    """
    try:
        logger.info(f"Loading matrix from {input_path}")
        T_group = np.loadtxt(input_path)
        logger.info(f"Loaded matrix shape: {T_group.shape}")
        if T_group.shape[0] < T_group.shape[1]:
            logger.info("Transposing matrix to shape (timepoints × components)")
            T_group = T_group.T
    except Exception as e:
        logger.exception(f"Failed to load input file: {e}")
        raise RuntimeError(f"Failed to load input file: {e}")

    logger.info(f"Running temporal ICA with model order k = {model_order}, seed = {seed}")
    try:
        M, B, tfm_ica = trifle_analysis.run_tICA(T_group, model_order, seed=seed)
    except Exception as e:
        logger.exception(f"Temporal ICA failed for k = {model_order}")
        raise RuntimeError(f"Temporal ICA failed for k = {model_order}: {e}")

    output_dir = Path(output_path) / f"tica_k{model_order}"
    output_dir.mkdir(parents=True, exist_ok=True)

    for arr, name in zip([M, B, tfm_ica], ["M", "B", "tica_model"]):
        out_file = output_dir / f"{name}.pkl"
        with open(out_file, "wb") as f_out:
            pickle.dump(arr, f_out)
        logger.info(f"Saved {name} to {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Run temporal ICA on spatial ICA time series.")
    parser.add_argument("--input", required=True, help="Path to the spatial ICA time series matrix (e.g., melodic_mix or DR output).")
    parser.add_argument("--output", required=True, help="Directory to save temporal ICA outputs.")
    parser.add_argument("--model_order", type=int, required=True, help="Model order (number of temporal components).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "trifle_temporal.log"

    # Configure logging only if not already set
    if not logger.hasHandlers():
        logging.basicConfig(filename=log_path, level=logging.INFO,
                            format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("===== Starting trifle_temporal.py =====")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output dir: {args.output}")
    logger.info(f"Model order: {args.model_order}")
    logger.info(f"Random seed: {args.seed}")

    run_temporal_ica(args.input, args.output, args.model_order, seed=args.seed)

    logger.info("===== trifle_temporal.py finished successfully =====")

if __name__ == "__main__":
    main()