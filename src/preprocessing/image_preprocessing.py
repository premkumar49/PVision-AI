"""
PVision AI: Image Preprocessing & Dataset Pipeline
Prepares PV fault image datasets for CNN training:
- Stratified Train / Validation / Test Splitting
- Image Resizing & Channel Normalization
- Data Augmentation Pipelines (Flip, Rotation, Jitter)
- PyTorch Dataset & DataLoader builder ready for manual training in VS Code
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from src.utilities.config import config
    from src.utilities.logger import get_logger
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from src.utilities.config import config
    from src.utilities.logger import get_logger

logger = get_logger("Image_Preprocessing")


class PVFaultDatasetManifest:
    """Creates stratified train, validation, and test splits for PV fault images."""

    def __init__(self, data_dir: Optional[Path] = None, modality: str = "gasf_images"):
        self.data_dir = Path(data_dir) if data_dir else (config.PV_FAULT_DATA_DIR / modality)
        self.modality = modality
        self.classes = config.PV_FAULT_CLASSES

    def create_split_manifest(
        self,
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = config.RANDOM_SEED
    ) -> Dict[str, pd.DataFrame]:
        """
        Builds stratified train/val/test dataframes with filepaths and class labels.
        """
        logger.info(f"Building dataset manifest for modality: {self.modality} from {self.data_dir}")
        records = []

        for cls_name in self.classes:
            cls_dir = self.data_dir / cls_name
            if not cls_dir.is_dir():
                continue
            for img_file in cls_dir.glob("*.png"):
                records.append({
                    "file_path": str(img_file),
                    "filename": img_file.name,
                    "class_name": cls_name,
                    "label": config.CLASS_TO_IDX[cls_name]
                })

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning(f"No image records found in {self.data_dir}")
            return {}

        logger.info(f"Loaded total {len(df):,} image references across {df['class_name'].nunique()} classes.")

        # Stratified train/temp split
        train_df, temp_df = train_test_split(
            df,
            test_size=(test_size + val_size),
            stratify=df["label"],
            random_state=random_state
        )

        # Stratified val/test split from temp
        rel_test_size = test_size / (test_size + val_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=rel_test_size,
            stratify=temp_df["label"],
            random_state=random_state
        )

        logger.info(f"Splits created: Train={len(train_df):,}, Val={len(val_df):,}, Test={len(test_df):,}")

        # Save manifests for reproducible training in VS Code
        out_dir = config.OUTPUTS_DIR / "predictions"
        out_dir.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(out_dir / f"{self.modality}_train_manifest.csv", index=False)
        val_df.to_csv(out_dir / f"{self.modality}_val_manifest.csv", index=False)
        test_df.to_csv(out_dir / f"{self.modality}_test_manifest.csv", index=False)
        logger.info(f"Manifest CSVs saved in {out_dir}")

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df
        }


def get_image_transforms(target_size: Tuple[int, int] = config.MODEL_INPUT_SIZE):
    """
    Returns image transformation pipelines for PyTorch / Torchvision.
    Ready for execution in VS Code when training models manually.
    """
    try:
        from torchvision import transforms

        train_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        eval_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return train_transform, eval_transform
    except ImportError:
        logger.warning("torchvision not installed in current environment. Returning placeholder dict.")
        return None, None
