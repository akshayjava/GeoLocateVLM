"""
Benchmark dataset loaders for GeoLocateVLM (Phase 4).

Available loaders
-----------------
im2gps3k  : load_im2gps3k()  — Im2GPS3k test set (Vo et al., 2017)
yfcc_val  : load_yfcc_val()  — YFCC100M streaming subset (HuggingFace)
            load_yfcc_from_csv() — offline CSV variant
"""
from src.benchmarks.im2gps3k import load_im2gps3k
from src.benchmarks.yfcc_val import load_yfcc_val, load_yfcc_from_csv, parse_yfcc_coords

__all__ = [
    "load_im2gps3k",
    "load_yfcc_val",
    "load_yfcc_from_csv",
    "parse_yfcc_coords",
]
