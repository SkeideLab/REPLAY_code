"""
paths.py
---------
Centralized file for handling all project paths.
Works consistently in both notebooks and scripts.

Usage:
    from src.utils.paths import DATA_RAW, DATA_PROCESSED
"""

from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
RAW_DIR = Path("path/to/raw/data")  # Replace with the actual path to your raw data
PREPROC_DIR = Path(
    "path/to/preprocessed/data"
)  # Replace with the actual path to your preprocessed data


# scripts and utils folders
UTILS_DIR = BASE_DIR / "utils"

# additional information and data directory
INFO_DIR = BASE_DIR / "additional_data"

# Graphics and results directories
FIGURES_DIR = Path("path/to/figures")  # Replace with the actual path to your figures
GRAPHICS_DIR = Path("path/to/graphics")  # Replace with the actual path to your graphics

# Ensure directories exist (optional safety)
# for path in [RAW_DIR, PREPROC_DIR]:
#    print(path.exists())
