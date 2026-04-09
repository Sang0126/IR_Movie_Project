# -*- coding: utf-8 -*-
"""
Module lam sach va chuan hoa du lieu (Data Cleaner).
- Tach biet viec xu ly van ban hien thi (UI) va van ban huan luyen (AI).
- Nhan trong so cac truong quan trong (Title, Director, Cast) de khong bi Overview lam lu mo.
- Loai bo that su cac gia tri "Unknown" khoi ma tran hoc may.
"""

import os
import sys
import pandas as pd
import ast
import re
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.config import PROCESSED_DATA_DIR, PROCESSED_MERGED

def parse_json_names(json_str, limit=None):
    if pd.isna(json_str) or json_str == '[]':
        return "Unknown"
    try:
        data = ast.literal_eval(json_str)
        if isinstance(data, list):
            names = [item.get('name', '') for item in data if isinstance(item, dict)]
            names = [n for n in names if n]
            if limit:
                names = names[:limit]
            if names:
                return ", ".join(names)
    except:
        pass
    return "Unknown"

def extract_director(crew_str):
    if pd.isna(crew_str) or crew_str == '[]':
        return "Unknown"
    try:
        crew = ast.literal_eval(crew_str)
        if isinstance(crew, list):
            for member in crew:
                if isinstance(member, dict) and member.get('job') == 'Director':
                    return member.get('name', 'Unknown')
    except:
        pass
    return "Unknown"

def sanitize_for_ui(text):
    """
    Giu nguyen dau cau chuc nang (! ? & : $) de hien thi Web dep mat.
    Nhung van loai bo ky tu tuong hinh khoi UI.
    """
    if pd.isna(text) or text == "":
        return ""
    text_str = str(text)
    text_str = unicodedata.normalize('NFKD', text_str).encode('ascii', 'ignore').decode('utf-8')
    return ' '.join(text_str.split())

def sanitize_for_ai(text):
    """
    Toi uu hoa cho mang TF-IDF.
    [NANG CAP]: Giu lai dau gach ngang (-) de bao toan cac tu ghep nhu sci-fi, spider-man, x-men...
    """
    if pd.isna(text) or text == "Unknown" or text == "":
        return ""
    text_str = str(text)
    cleaned = re.sub(r'[^a-zA-Z0-9\s\-]', ' ', text_str)
    return ' '.join(cleaned.split()).lower()

def format_entity(text, prefix):
    """Dong goi the Entity Encoding dac quyen, bo qua Unknown"""
    if not text or text == "Unknown":
        return ""
    items = str(text).split(',')
    formatted = [f"{prefix}_{sanitize_for_ai(item).replace(' ', '_').replace('-', '_')}" for item in items if item.strip()]
    return " ".join(formatted)

def clean_data():
    input_path = PROCESSED_MERGED
    if not os.path.exists(input_path):
        print(f"[ERROR] Khong tim thay du lieu gop tai: {input_path}")
        return False
        
    print("\n" + "="*80)
    print("BAT DAU QUY TRINH THANH LOC DU LIEU (STANDARD DATA ENGINEERING)")
    print("="*80)
    
    df = pd.read_csv(input_path, low_memory=False, encoding='utf-8')
    initial_rows = len(df)
    print(f"[INFO] Trang thai ban dau: nap {initial_rows:,} dong du lieu.")

    removed_records = [] 

    # 1. LOAI BO TRUNG LAP VA THIEU ID
    if 'movieId' in df.columns:
        duplicates = df[df.duplicated(subset=['movieId'], keep='first')].copy()
        if not duplicates.empty:
            duplicates['reason_removed'] = "Trung lap movieId"
            removed_records.append(duplicates)
        df = df.drop_duplicates(subset=['movieId'], keep='first')

    if 'tmdbId' in df.columns:
        missing_id = df[df['tmdbId'].isna()].copy()
        if not missing_id.empty:
            missing_id['reason_removed'] = "Thieu tmdbId"
            removed_records.append(missing_id)
        df = df.dropna(subset=['tmdbId'])
        df['tmdbId'] = df['tmdbId'].astype(float).astype(int)

    # 2. TACH TEN PHIM
    possible_title_cols = ['title', 'original_title', 'title_x', 'title_y', 'Title']
    title_col = next((c for c in possible_title_cols if c in df.columns), None)
    
    if title_col:
        missing_title = df[df[title_col].isna() | (df[title_col].astype(str).str.strip() == '')].copy()
        if not missing_title.empty:
            missing_title['reason_removed'] = "Thieu ten phim"
            removed_records.append(missing_title)
            
        df = df.dropna(subset=[title_col])
        df = df[df[title_col].astype(str).str.strip() != '']

        extracted = df[title_col].str.extract(r'^(.*?)\s*\((\d{4})\)\s*$')
        df['title_clean'] = extracted[0].fillna(df[title_col]) 
        
        if 'release_date' in df.columns:
            tmdb_year = pd.to_datetime(df['release_date'], errors='coerce').dt.year
            df['release_year'] = extracted[1].fillna(tmdb_year).fillna(0).astype(int)
        else:
            df['release_year'] = extracted[1].fillna(0).astype(int)

    # 3. BOC TACH VA CHUAN HOA GIAO DIEN (UI)
    print("[INFO] Dang boc tach Json va chuan hoa hien thi Web (Giu lai dau cau)...")
    if 'genres' in df.columns and ('{' in str(df['genres'].iloc[0]) or '[' in str(df['genres'].iloc[0])):
        df['genres_str'] = df['genres'].apply(parse_json_names)
    elif 'genres_y' in df.columns:
        df['genres_str'] = df['genres_y'].apply(parse_json_names)
    elif 'genres_tmdb' in df.columns:
        df['genres_str'] = df['genres_tmdb'].apply(parse_json_names)
    elif 'genres' in df.columns:
        df['genres_str'] = df['genres'].fillna("Unknown").apply(lambda x: str(x).replace('|', ', '))
    else:
        df['genres_str'] = "Unknown"

    df['cast_str'] = df['cast'].apply(lambda x: parse_json_names(x, limit=5)) if 'cast' in df.columns else "Unknown"
    df['director'] = df['crew'].apply(extract_director) if 'crew' in df.columns else "Unknown"
    df['keywords_str'] = df['keywords'].apply(parse_json_names) if 'keywords' in df.columns else ""
    df['overview'] = df.get('overview', pd.Series([""]*len(df))).fillna("")
    df['user_tags'] = df.get('user_tags', pd.Series([""]*len(df))).fillna("")

    text_columns_ui = ['title_clean', 'genres_str', 'cast_str', 'director', 'keywords_str', 'overview', 'user_tags']
    for col in text_columns_ui:
        df[col] = df[col].apply(sanitize_for_ui)
        df[col] = df[col].replace("", "Unknown")

    # 4. LOC PHIM CHET KHOI DATA
    df['title_clean'] = df['title_clean'].astype(str).str.strip()
    
    def is_valid_title(t):
        return bool(re.search(r'[a-zA-Z0-9]', str(t)))

    invalid_movies = df[~df['title_clean'].apply(is_valid_title)].copy()
    if not invalid_movies.empty:
        invalid_movies['reason_removed'] = "Ten phim khong con chua the chu cai/so hoc"
        removed_records.append(invalid_movies)
        
    df = df[df['title_clean'].apply(is_valid_title)]

    if 'poster_path' in df.columns:
        df['poster_path'] = df['poster_path'].fillna("").apply(
            lambda x: str(x) if str(x).startswith('/') else ('/' + str(x) if str(x) and str(x) not in ['nan', 'None', 'Unknown'] else "")
        )
    else:
        df['poster_path'] = ""

    # 5. TINH TOAN DIEM PHO BIEN IMDB
    print("[INFO] Dang tinh toan Diem Pho Bien (IMDB Weighted Rating)...")
    df['avg_rating'] = df.get('avg_rating', pd.Series([0.0]*len(df))).fillna(0.0)
    df['num_ratings'] = df.get('num_ratings', pd.Series([0]*len(df))).fillna(0).astype(int)
    df['runtime'] = df.get('runtime', pd.Series([0]*len(df))).fillna(0).astype(int)

    valid_votes = df[df['num_ratings'] > 0]
    mean_rating = valid_votes['avg_rating'].mean() if not valid_votes.empty else 0.0
    min_votes = valid_votes['num_ratings'].quantile(0.70) if not valid_votes.empty else 0.0

    v = df['num_ratings']
    R = df['avg_rating']
    m = min_votes
    C = mean_rating

    df['popularity_score'] = 0.0
    mask = (v > 0)
    df.loc[mask, 'popularity_score'] = ((v[mask] / (v[mask] + m)) * R[mask]) + ((m / (v[mask] + m)) * C)
    df['popularity_score'] = df['popularity_score'].round(3)

    # 6. TAO COT 'CONTENT' CHO AI VOI [TRONG SO - WEIGHTING] BOUNG CAC TRUONG QUAN TRONG
    print("[INFO] Dang ep Ma tran NLP (Nhan Trong So cho Title, Cast, Director de ap dao Overview)...")
    
    def safe_ai_text(series):
        return series.replace("Unknown", "").apply(sanitize_for_ai)

    ai_title = safe_ai_text(df['title_clean'])
    ai_genres = safe_ai_text(df['genres_str'])
    ai_director_raw = safe_ai_text(df['director'])
    ai_cast_raw = safe_ai_text(df['cast_str'])
    ai_kw = safe_ai_text(df['keywords_str'])
    ai_tags = safe_ai_text(df['user_tags'])
    ai_overview = safe_ai_text(df['overview'])

    # [NÂNG CẤP]: Cụm text nhân trọng số để định hướng TF-IDF tập trung vào Entity
    weighted_title = (ai_title + " ") * 3
    weighted_genres = (ai_genres + " " + df['genres_str'].apply(lambda x: format_entity(x, 'genre')) + " ") * 2
    weighted_director = (ai_director_raw + " " + df['director'].apply(lambda x: format_entity(x, 'dir')) + " ") * 2
    weighted_cast = (ai_cast_raw + " " + df['cast_str'].apply(lambda x: format_entity(x, 'cast')) + " ") * 2
    weighted_kw = (ai_kw + " " + df['keywords_str'].apply(lambda x: format_entity(x, 'kw')) + " ") * 2
    weighted_tags = (ai_tags + " " + df['user_tags'].apply(lambda x: format_entity(x, 'tag')) + " ") * 2

    # Overview chỉ giữ hệ số x1 để cung cấp ngữ nghĩa nền tảng mà không nuốt chửng Entity
    df['content'] = (
        weighted_title + 
        weighted_genres + 
        weighted_director + 
        weighted_cast + 
        weighted_kw + 
        weighted_tags + 
        ai_overview
    )
    df['content'] = df['content'].apply(lambda x: " ".join(str(x).split()).lower())

    # 7. LUU DU LIEU
    columns_to_keep = [
        'movieId', 'tmdbId', 'title_clean', 'release_year', 'director', 
        'cast_str', 'genres_str', 'runtime', 'avg_rating', 'num_ratings', 
        'popularity_score', 'overview', 'poster_path', 'content'
    ]
    final_cols = [c for c in columns_to_keep if c in df.columns]
    df_final = df[final_cols]
    
    out_path = os.path.join(PROCESSED_DATA_DIR, "movies_processed.csv")
    df_final.to_csv(out_path, index=False, encoding='utf-8-sig') 
    
    removed_path = os.path.join(PROCESSED_DATA_DIR, "movies_removed.csv")
    if removed_records:
        pd.concat(removed_records, ignore_index=True).to_csv(removed_path, index=False, encoding='utf-8-sig')
    else:
        pd.DataFrame(columns=['reason_removed']).to_csv(removed_path, index=False)
    
    print("\n[REPORT] HOAN TAT CHUAN HOA DU LIEU:")
    print(f" -> Giao dien Web    : Dấu câu (!,&,?) được bảo vệ thành công.")
    print(f" -> Trí tuệ AI       : Các trường MetaData mấu chốt được nhân Trọng số x2, x3.")
    print(f" -> Tong so bo phim  : {len(df_final):,} bộ.")
    print("="*80 + "\n")
    return True

if __name__ == "__main__":
    clean_data()