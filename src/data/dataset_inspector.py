"""
PVision AI: Master Dataset Inspection Runner
Unified CLI and programmatic interface for inspecting both PV Fault and Solar Generation datasets.
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utilities.config import config
from src.utilities.logger import get_logger
from src.data.inspect_pv_fault import PVFaultInspector
from src.data.inspect_solar_generation import SolarGenerationInspector

logger = get_logger("Master_Dataset_Inspector")


def run_master_inspection(
    inspect_pv: bool = True,
    inspect_solar: bool = True,
    check_duplicates: bool = True,
    max_duplicate_samples: int = 3500
):
    """Executes dataset inspection across selected domains."""
    config.ensure_directories()
    logger.info("Initializing PVision AI Dataset Inspection Suite...")

    master_results = {
        "project": "PVision AI",
        "description": "Hybrid CNN-ANN Framework for Photovoltaic Fault Detection and Power Generation Prediction",
        "pv_fault": None,
        "solar_generation": None
    }

    if inspect_pv:
        logger.info("\n>>> EXECUTING PV FAULT DATASET INSPECTION <<<")
        pv_inspector = PVFaultInspector()
        master_results["pv_fault"] = pv_inspector.inspect(
            check_duplicates=check_duplicates,
            max_duplicate_samples=max_duplicate_samples
        )

    if inspect_solar:
        logger.info("\n>>> EXECUTING SOLAR GENERATION DATASET INSPECTION <<<")
        solar_inspector = SolarGenerationInspector()
        master_results["solar_generation"] = solar_inspector.inspect()

    # Save Master Executive Summary
    master_path = config.METRICS_DIR / "pvision_master_inspection_summary.json"
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master_results, f, indent=4)
    logger.info(f"\nSuccessfully generated Master Inspection Summary at: {master_path}")

    return master_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PVision AI Master Dataset Inspector")
    parser.add_argument("--all", action="store_true", default=True, help="Run inspection on both datasets")
    parser.add_argument("--pv-only", action="store_true", help="Inspect only PV fault images")
    parser.add_argument("--solar-only", action="store_true", help="Inspect only Solar generation CSVs")
    parser.add_argument("--no-duplicates", action="store_true", help="Skip MD5 duplicate image scan")
    parser.add_argument("--sample-limit", type=int, default=3000, help="Sample limit for duplicate checks")

    args = parser.parse_args()

    inspect_pv = not args.solar_only
    inspect_solar = not args.pv_only

    run_master_inspection(
        inspect_pv=inspect_pv,
        inspect_solar=inspect_solar,
        check_duplicates=not args.no_duplicates,
        max_duplicate_samples=args.sample_limit
    )
