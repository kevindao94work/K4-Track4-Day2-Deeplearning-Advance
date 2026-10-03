"""Đường dẫn dùng chung; tách lần tái lập khỏi sản phẩm đã nộp bằng LAB_OUTPUT_DIR."""
import os
from pathlib import Path

CODE=Path(__file__).resolve().parent
REPO=next(p for p in CODE.parents if (p/'eval.py').is_file())
SUB=Path(os.environ.get('LAB_OUTPUT_DIR',str(CODE.parent))).resolve()
DATA=Path(os.environ.get('LAB_DATA_DIR',str(REPO/'data'))).resolve()
