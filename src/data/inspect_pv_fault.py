"""
PVision AI: PV Fault Dataset Inspector
Performs in-depth analysis on PV fault image datasets (GASF and I-V curves):
- Image discovery and file identification
- Label and class verification
- Sample counting and class distribution analysis
- Missing sequence file detection and corruption verification
- Duplicate image detection via cryptographic hashing (MD5)
- Automated summary export (JSON) and distribution visualization (PNG)
"""

import os
import sys
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt

# Handle imports when run directly or as a module
try:
    from src.utilities.config import config
    from src.utilities.logger import get_logger
except ImportError:
    # If run from within src/data or direct path
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from src.utilities.config import config
    from src.utilities.logger import get_logger

logger = get_logger("PV_Fault_Inspector")


class PVFaultInspector:
    """Inspector for Photovoltaic Fault Image Datasets."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir) if data_dir else config.PV_FAULT_DATA_DIR
        self.expected_classes = config.PV_FAULT_CLASSES
        self.modalities = config.MODALITIES
        self.summary_report: Dict[str, Any] = {}

    def inspect(self, check_duplicates: bool = True, max_duplicate_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Executes full inspection workflow.

        Args:
            check_duplicates: Whether to compute MD5 hashes for duplicate detection.
            max_duplicate_samples: Optional cap on files checked for duplicate hashing.
        """
        logger.info(f"Starting PV Fault Dataset Inspection at: {self.data_dir}")
        config.ensure_directories()

        if not self.data_dir.exists():
            logger.error(f"PV Fault directory does not exist: {self.data_dir}")
            return {"error": f"Directory not found: {self.data_dir}"}

        modalities_found = [m for m in self.modalities if (self.data_dir / m).is_dir()]
        if not modalities_found:
            # Check if classes are directly in data_dir (flat structure)
            direct_classes = [c for c in self.expected_classes if (self.data_dir / c).is_dir()]
            if direct_classes:
                modalities_found = ["."]
            else:
                logger.warning(f"No standard modalities or classes found in {self.data_dir}")

        logger.info(f"Found modalities/directories to inspect: {modalities_found}")

        overall_summary = {
            "dataset_path": str(self.data_dir),
            "modalities_inspected": modalities_found,
            "total_images": 0,
            "modality_reports": {}
        }

        total_images_all = 0

        for modality in modalities_found:
            mod_path = self.data_dir if modality == "." else self.data_dir / modality
            mod_report = self._inspect_modality(
                modality_name=modality,
                modality_path=mod_path,
                check_duplicates=check_duplicates,
                max_samples=max_duplicate_samples
            )
            overall_summary["modality_reports"][modality] = mod_report
            total_images_all += mod_report["total_images"]

        overall_summary["total_images"] = total_images_all
        self.summary_report = overall_summary

        # Export outputs
        self._export_summary_json()
        self._generate_visualizations()
        self._print_console_summary()

        return overall_summary

    def _inspect_modality(
        self,
        modality_name: str,
        modality_path: Path,
        check_duplicates: bool = True,
        max_samples: Optional[int] = None
    ) -> Dict[str, Any]:
        """Inspects an individual modality folder (e.g., gasf_images or iv_images)."""
        logger.info(f"--- Inspecting Modality: {modality_name} ---")

        classes_found = []
        class_counts = {}
        missing_classes = []
        unexpected_classes = []
        image_properties = defaultdict(set)
        corrupted_files = []
        missing_sequence_files = {}

        # 1. Class Discovery
        subdirs = [d.name for d in modality_path.iterdir() if d.is_dir()]
        for exp_cls in self.expected_classes:
            if exp_cls in subdirs:
                classes_found.append(exp_cls)
            else:
                missing_classes.append(exp_cls)

        for d in subdirs:
            if d not in self.expected_classes:
                unexpected_classes.append(d)

        # 2. File and Sample Inspection
        total_modality_images = 0
        file_list = []
        sample_indices_per_class = defaultdict(list)

        for cls_name in classes_found:
            cls_dir = modality_path / cls_name
            cls_files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in config.EXPECTED_IMAGE_EXTS]
            count = len(cls_files)
            class_counts[cls_name] = count
            total_modality_images += count

            # Extract sequence numbers from filenames if pattern matches (e.g. crack_00001_GASF.png)
            extracted_indices = []
            for f in cls_files:
                file_list.append((f, cls_name))
                # Search for digits in filename
                match = re.search(r"(\d+)", f.stem)
                if match:
                    extracted_indices.append(int(match.group(1)))

            sample_indices_per_class[cls_name] = sorted(extracted_indices)

        # 3. Detect Missing Sequence Files
        for cls_name, indices in sample_indices_per_class.items():
            if indices:
                min_idx = min(indices)
                max_idx = max(indices)
                # If sequence appears to start around 1 and max is around expected 5000
                expected_max = 5000 if max_idx <= 5000 and max_idx > 3000 else max_idx
                expected_range = set(range(1, expected_max + 1))
                actual_set = set(indices)
                missing = sorted(list(expected_range - actual_set))

                if missing:
                    missing_sequence_files[cls_name] = {
                        "expected_count": expected_max,
                        "present_count": len(indices),
                        "missing_count": len(missing),
                        "missing_index_ranges": self._compress_ranges(missing),
                        "sample_missing_indices": missing[:10]
                    }

        # 4. Check Image Properties & Corruption on a representative sample
        # Sample up to 100 images per class for detailed format & integrity checks
        checked_for_corruption = 0
        for cls_name in classes_found:
            cls_dir = modality_path / cls_name
            sample_files = [f for f in cls_dir.iterdir() if f.is_file() and f.suffix.lower() in config.EXPECTED_IMAGE_EXTS][:100]
            for img_file in sample_files:
                checked_for_corruption += 1
                try:
                    if img_file.stat().st_size == 0:
                        corrupted_files.append((str(img_file), "Zero-byte file"))
                        continue
                    with Image.open(img_file) as img:
                        img.verify()  # Fast structural verification
                    # Re-open for dimensions and mode
                    with Image.open(img_file) as img:
                        image_properties["sizes"].add(img.size)
                        image_properties["modes"].add(img.mode)
                        image_properties["formats"].add(img.format)
                except Exception as e:
                    corrupted_files.append((str(img_file), str(e)))

        # 5. Duplicate Image Detection (Hash-based)
        duplicate_report = {"duplicates_found": 0, "duplicate_clusters": []}
        if check_duplicates and file_list:
            logger.info(f"Checking for duplicates in {modality_name}...")
            duplicate_report = self._detect_duplicates(file_list, max_samples=max_samples)

        # 6. Class Distribution & Balance Calculation
        distribution_pct = {}
        for cls_name, count in class_counts.items():
            pct = (count / total_modality_images * 100) if total_modality_images > 0 else 0
            distribution_pct[cls_name] = round(pct, 2)

        return {
            "modality": modality_name,
            "total_images": total_modality_images,
            "classes_identified": classes_found,
            "class_counts": class_counts,
            "class_distribution_percentage": distribution_pct,
            "missing_classes": missing_classes,
            "unexpected_classes": unexpected_classes,
            "image_resolutions_detected": [list(s) for s in image_properties.get("sizes", set())],
            "color_modes_detected": list(image_properties.get("modes", set())),
            "formats_detected": list(image_properties.get("formats", set())),
            "missing_sequence_details": missing_sequence_files,
            "corrupted_files_count": len(corrupted_files),
            "corrupted_files_sample": corrupted_files[:5],
            "duplicates": duplicate_report
        }

    def _detect_duplicates(self, file_list: List[tuple], max_samples: Optional[int] = None) -> Dict[str, Any]:
        """Detects duplicate files using fast MD5 file hashing."""
        hashes = defaultdict(list)
        files_to_check = file_list[:max_samples] if max_samples else file_list

        for idx, (file_path, cls_name) in enumerate(files_to_check):
            try:
                # Fast chunked MD5 calculation
                hasher = hashlib.md5()
                with open(file_path, "rb") as f:
                    # Read first 64KB for speed, or entire file for small images
                    chunk = f.read(65536)
                    while chunk:
                        hasher.update(chunk)
                        chunk = f.read(65536)
                file_hash = hasher.hexdigest()
                hashes[file_hash].append((str(file_path.name), cls_name))
            except Exception:
                continue

        duplicates = {h: files for h, files in hashes.items() if len(files) > 1}
        num_duplicates = sum(len(files) - 1 for files in duplicates.values())

        sample_clusters = []
        for h, files in list(duplicates.items())[:5]:
            sample_clusters.append({
                "hash": h,
                "count": len(files),
                "files": files[:3]
            })

        return {
            "samples_checked": len(files_to_check),
            "duplicates_found": num_duplicates,
            "duplicate_groups": len(duplicates),
            "sample_clusters": sample_clusters
        }

    @staticmethod
    def _compress_ranges(nums: List[int]) -> List[str]:
        """Converts a sorted list of numbers into contiguous string ranges (e.g. 1-513)."""
        if not nums:
            return []
        ranges = []
        start = nums[0]
        prev = nums[0]

        for n in nums[1:]:
            if n == prev + 1:
                prev = n
            else:
                ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
                start = n
                prev = n
        ranges.append(f"{start}-{prev}" if start != prev else f"{start}")
        return ranges

    def _export_summary_json(self):
        """Saves inspection findings as a structured JSON artifact."""
        output_file = config.METRICS_DIR / "pv_fault_dataset_summary.json"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.summary_report, f, indent=4)
            logger.info(f"Saved PV fault dataset summary to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to export summary JSON: {e}")

    def _generate_visualizations(self):
        """Generates class distribution bar charts for inspected modalities."""
        mod_reports = self.summary_report.get("modality_reports", {})
        if not mod_reports:
            return

        fig, axes = plt.subplots(1, len(mod_reports), figsize=(7 * len(mod_reports), 5), squeeze=False)
        colors = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]

        for idx, (mod_name, report) in enumerate(mod_reports.items()):
            ax = axes[0, idx]
            class_counts = report.get("class_counts", {})
            classes = list(class_counts.keys())
            counts = list(class_counts.values())

            bars = ax.bar(classes, counts, color=colors[:len(classes)], edgecolor="black", alpha=0.85)
            ax.set_title(f"Class Distribution: {mod_name}\nTotal: {report.get('total_images', 0):,} Images", fontsize=12, fontweight="bold")
            ax.set_ylabel("Number of Samples", fontsize=10)
            ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.5)

            # Annotate bar counts
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:,}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.tight_layout()
        out_path = config.FIGURES_DIR / "pv_fault_class_distribution.png"
        try:
            plt.savefig(out_path, dpi=200)
            plt.close()
            logger.info(f"Saved class distribution figure to: {out_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {e}")

    def _print_console_summary(self):
        """Prints a human-readable table report to stdout."""
        print("\n" + "=" * 70)
        print("           PV FAULT DATASET INSPECTION REPORT")
        print("=" * 70)
        print(f"Path: {self.data_dir}")
        print(f"Total Images Across All Modalities: {self.summary_report.get('total_images', 0):,}")
        print("-" * 70)

        for mod_name, rep in self.summary_report.get("modality_reports", {}).items():
            print(f"\n[Modality: {mod_name}]")
            print(f"Total Samples: {rep.get('total_images', 0):,}")
            print(f"Resolutions: {rep.get('image_resolutions_detected')}")
            print(f"Color Modes: {rep.get('color_modes_detected')}")
            print(f"Corrupted Files Detected: {rep.get('corrupted_files_count')}")

            print("\n  Class Distribution Breakdown:")
            print(f"  {'Class Name':<20} | {'Count':<10} | {'Percentage':<10}")
            print("  " + "-" * 46)
            for cls_name, count in rep.get("class_counts", {}).items():
                pct = rep.get("class_distribution_percentage", {}).get(cls_name, 0.0)
                print(f"  {cls_name:<20} | {count:<10,} | {pct:>6.2f}%")

            missing_seq = rep.get("missing_sequence_details", {})
            if missing_seq:
                print("\n  Missing Sequence Detection:")
                for cls_name, info in missing_seq.items():
                    print(f"  - {cls_name}: {info['missing_count']} missing files (Ranges: {', '.join(info['missing_index_ranges'])})")

            dup_info = rep.get("duplicates", {})
            if dup_info:
                print(f"\n  Duplicate Detection (Checked {dup_info.get('samples_checked', 0):,} files):")
                print(f"  Duplicates Found: {dup_info.get('duplicates_found', 0)}")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    inspector = PVFaultInspector()
    inspector.inspect(check_duplicates=True, max_duplicate_samples=2000)
