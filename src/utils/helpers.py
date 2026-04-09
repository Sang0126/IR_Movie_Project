"""Các hàm tiện ích hỗ trợ."""

import pandas as pd
import numpy as np
import json
import ast
import re
from typing import List, Dict, Any


def safe_literal_eval(val):
    """
    Chuyển đổi chuỗi thành list hoặc dict một cách an toàn.
    Vi du: "[1, 2, 3]" -> [1, 2, 3]
    """
    try:
        return ast.literal_eval(val) if isinstance(val, str) else val
    except (ValueError, SyntaxError):
        return []


def parse_json_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Xử lý cột chứa dữ liệu JSON dạng chuỗi thành cấu trúc dữ liệu Python (list hoặc dict).
    Tham số:
        df: DataFrame chứa cột cần xử lý
        column: Tên cột cần xử lý
    Trả về:
        Nếu đúng: DataFramd đã được xử lý. 
        Nếu sai: trả về list rỗng.
    """
    if column not in df.columns:
        return df
    
    df[column] = df[column].fillna('[]')
    df[column] = df[column].apply(safe_literal_eval)
    return df


def extract_names_from_json(json_list: List[Dict], key: str = 'name') -> List[str]:
    """
    Trich xuat truong 'name' tu danh sach cac dictionary.
    
    Vi du:
        Input: [{'id': 1, 'name': 'Action'}, {'id': 2, 'name': 'Comedy'}]
        Output: ['Action', 'Comedy']
    
    Tham so:
        json_list: Danh sach cac dictionary
        key: Ten truong can trich xuat (mac dinh la 'name')
    
    Tra ve:
        Danh sach cac gia tri
    """
    if not isinstance(json_list, list):
        return []
    return [item[key] for item in json_list if isinstance(item, dict) and key in item]


def clean_text(text: str) -> str:
    """
    Lam sach van ban de xu ly.
    
    Cac buoc:
    1. Chuyen thanh chu thuong
    2. Loai bo ky tu dac biet
    3. Loai bo khoang trang thua
    
    Tham so:
        text: Van ban can lam sach
    
    Tra ve:
        Van ban da lam sach
    """
    if not isinstance(text, str):
        return ""
    
    # Chuyen thanh chu thuong
    text = text.lower()
    
    # Chi giu lai chu cai, so va khoang trang
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Loai bo khoang trang thua
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def combine_features(row: pd.Series, features: List[str], separator: str = ' ') -> str:
    """
    Ket hop nhieu truong van ban thanh mot chuoi.
    
    Vi du: Ket hop 'title', 'overview', 'genres' thanh mot chuoi de index cho tim kiem
    
    Tham so:
        row: Dong du lieu tu DataFrame
        features: Danh sach ten cot can ket hop
        separator: Ky tu phan cach (mac dinh la khoang trang)
    
    Tra ve:
        Chuoi da ket hop
    """
    texts = []
    for feature in features:
        if feature in row and pd.notna(row[feature]):
            val = row[feature]
            if isinstance(val, list):
                val = ' '.join(str(v) for v in val)
            texts.append(str(val))
    
    return separator.join(texts)


def format_runtime(minutes: float) -> str:
    """Chuyen doi thoi luong phim tu phut sang dinh dang 'Xh Ym'."""
    if pd.isna(minutes) or minutes == 0:
        return "Khong co thong tin"
    
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    
    if hours > 0:
        return f"{hours} gio {mins} phut"
    return f"{mins} phut"


def format_rating(rating: float) -> str:
    """Dinh dang diem danh gia."""
    if pd.isna(rating):
        return "Khong co danh gia"
    return f"{rating:.1f}/10"


def get_year_from_date(date_str: str) -> int:
    """
    Trich xuat nam tu chuoi ngay thang.
    
    Vi du: "2023-12-25" -> 2023
    Trả về NAN nếu không thể trích xuất được năm.
    """
    if pd.isna(date_str) or not isinstance(date_str, str):
        return None
    
    # Tim so co 4 chu so bat dau bang 19 hoac 20 (nam)
    match = re.search(r'\b(19|20)\d{2}\b', date_str)
    if match:
        return int(match.group())
    return None


def save_pickle(obj: Any, filepath: str):
    """Luu doi tuong vao file pickle."""
    import pickle
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Da luu vao {filepath}")


def load_pickle(filepath: str) -> Any:
    """Doc doi tuong tu file pickle."""
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)


