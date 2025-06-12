#!/usr/bin/env python

import argparse
import logging
import shutil
import subprocess
from pathlib import Path
import trifle_module.templates

# ---------- Set up module-level logger ----------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True

def run_spatial_ica(input_files, output_dir, tr, n_components=20, bet=False, mask=None):
    logger.info("Running group spatial ICA with parameters:")
    logger.info(f"  Input files: {input_files}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  TR: {tr}")
    logger.info(f"  Components: {n_components}")
    logger.info(f"  BET: {bet}")
    logger.info(f"  Mask: {mask}")

    cmd = [
        "melodic",
        "-i", input_files,
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

def get_template_path(template_name):
    import importlib.resources as pkg_resources
    template_map = {
        "smith20": "Smith20.nii.gz",
        "smith70": "Smith70.nii.gz"
    }
    if template_name not in template_map:
        raise ValueError(f"Unknown template: {template_name}")
    return str(pkg_resources.files(trifle_module.templates) / template_map[template_name])

def run_dual_regression(template_name, input_data, output_dir):
    logger.info("Running dual regression with parameters:")
    logger.info(f"  Template: {template_name}")
    logger.info(f"  Input data: {input_data}")
    logger.info(f"  Output dir: {output_dir}")

    if shutil.which("dual_regression") is None:
        raise RuntimeError("FSL's 'dual_regression' script not found in PATH.")

    template_path = get_template_path(template_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "dual_regression",
        str(template_path),
        "1",      # variance-normalise timecourses
        "-1",     # skip design files
        "0",      # no permutations
        str(output_dir),
        str(input_data)
    ]
    logger.info("Executing dual_regression command:")
    logger.info("  " + " ".join(command))
    subprocess.run(command, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run spatial ICA or dual regression using TRIFLE.")
    parser.add_argument("--level", choices=["group", "individual"], required=True)
    parser.add_argument("--input", required=True, help="Path to 4D preprocessed functional data (.nii.gz).")
    parser.add_argument("--output", required=True, help="Directory for outputs.")
    parser.add_argument("--tr", required=True, type=float, help="Repetition time in seconds.")
    parser.add_argument("--n_components", type=int, default=20)
    parser.add_argument("--bet", action="store_true")
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--template", choices=["smith20", "smith70"], default="smith20")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    logfile_path = output_dir / "trifle_spatial.log"
    logging.basicConfig(filename=logfile_path, filemode='w', level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    logger.info("===== Starting trifle_spatial.py =====")

    if args.level == "group":
        run_spatial_ica(args.input, args.output, args.tr, args.n_components, args.bet, args.mask)
    elif args.level == "individual":
        run_dual_regression(args.template, args.input, args.output)

    logger.info("===== trifle_spatial.py finished successfully =====")

if __name__ == "__main__":
    main()