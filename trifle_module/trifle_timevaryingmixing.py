#!/usr/bin/env python

# TRIFLE - TIME-VARYING MIXING COMPUTATION
# Version: 11-06-2025 by TJ de Kloe 

import argparse
import pickle
import logging
from pathlib import Path
import trifle_module.trifle_analysis as trifle_analysis
import trifle_module.trifle_dataload as trifle_dataload

# ---------- Logger setup ----------
logger                  = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate        = True

def run_trifle_analysis(input_file, S_file, T_file, M_file, B_file, mask_file, output_dir):
    """
    Run TRIFLE core analysis: compute reconstructed signal and time-varying mixing matrix f.
    """
    logger.info("===== Starting TRIFLE core analysis =====")

    # Load data
    data2d, _           = trifle_dataload.load_X(input_file)
    spatialmap2d, _     = trifle_dataload.load_S(S_file)
    brain_voxels, X, X_shape, S, S_shape = trifle_dataload.mask(mask_file, data2d, spatialmap2d)
    T, Tz, M, B, Bz     = trifle_dataload.load_TMB(T_file, M_file, B_file)

    # Compute time-varying mixing
    Xr, M_t, Mtsum, f   = trifle_analysis.compute_tvM(S, M, Bz)

    logger.info("TRIFLE computation completed successfully.")
    logger.info(f"  Reconstructed signal shape: {Xr.shape}")
    logger.info(f"  Temporal weighting matrix f shape: {f.shape}")

    # Save outputs
    output_dir          = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "Xr.pkl", "wb") as f_out:
        pickle.dump(Xr, f_out)
    with open(output_dir / "f.pkl", "wb") as f_out:
        pickle.dump(f, f_out)
    with open(output_dir / "Mtsum.pkl", "wb") as f_out:
        pickle.dump(Mtsum, f_out)

    logger.info(f"Outputs saved in {output_dir}")

def main():
    parser              = argparse.ArgumentParser(
        description="Run TRIFLE final step: compute reconstructed signal and time-varying mixing matrix f."
    )
    parser.add_argument("--input", required=True, help="4D functional input data (e.g., group-level .nii.gz or list file).")
    parser.add_argument("--S", required=True, help="Path to spatial ICA map (melodic_IC.nii.gz).")
    parser.add_argument("--T", required=True, help="Path to spatial time series file (melodic_mix).")
    parser.add_argument("--M", required=True, help="Path to M.pkl (temporal mixing matrix).")
    parser.add_argument("--B", required=True, help="Path to B.pkl (temporal source matrix).")
    parser.add_argument("--mask", required=True, help="Path to brain mask file.")
    parser.add_argument("--output", required=True, help="Output directory for final results.")

    args                = parser.parse_args()

    output_dir          = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path            = output_dir / "trifle_timevaryingmixing.log"
    logging.basicConfig(filename=log_path, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    run_trifle_analysis(
        input_file=args.input,
        S_file=args.S,
        T_file=args.T,
        M_file=args.M,
        B_file=args.B,
        mask_file=args.mask,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()