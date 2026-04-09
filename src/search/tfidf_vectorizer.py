"""Mô đun xây dựng cấu trúc Ma trận Truy hồi (Vectorizing Pipeline)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize # Hỗ trợ chuẩn hóa L2 cho Matrix LSA

from src.utils.config import PROCESSED_DATA_DIR, TFIDF_MATRIX_PATH


def build_search_matrix():
    print("\n" + "="*80)
    print("BẮT ĐẦU CHƯƠNG TRÌNH HUẤN LUYỆN MÔ HÌNH NGÔN NGỮ (TF-IDF & LSA)")
    print("="*80)
    
    print("[INFO] Đang nạp hệ cơ sở dữ liệu xử lý vào RAM...")
    data_path = os.path.join(PROCESSED_DATA_DIR, "movies_processed.csv")
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Không tìm thấy dữ liệu đã làm sạch ở: {data_path}")
        return False
        
    df = pd.read_csv(data_path, low_memory=False)
    print(f"[INFO] Hệ thống ghi nhận {len(df):,} tài liệu (Documents).\n")
    
    missing = df['content'].isna().sum()
    if missing > 0:
        df['content'] = df['content'].fillna('')
        
    documents = df['content'].tolist()

    # KHỞI TẠO BƯỚC 1: TF-IDF
    print("[STEP 1] KHỞI TẠO MA TRẬN TẦN SUẤT - NGHỊCH ĐẢO TÀI LIỆU (TF-IDF)")
    print(" -> Đang phân tích trọng số Từ vựng (Vocabulary Building)...")
    
    # NÂNG CẤP VECTORIZER: Dùng min_df/max_df chống Noise và Tăng 15,000 vocab
    vectorizer = TfidfVectorizer(
        max_features=15000,   
        stop_words='english', 
        ngram_range=(1, 2),
        min_df=2,             # Bỏ qua từ vựng xuất hiện duy nhất 1 lần (Thường là lỗi chính tả)
        max_df=0.85           # Bỏ qua từ vựng xuất hiện ở 85% tổng số phim (Quá chung chung như "movie", "film")
    )
    
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print(f" -> Kích thước tập từ vựng chuẩn (Vocab Size): {len(vectorizer.vocabulary_):,} từ vựng.")
    print(f" -> Kích thước Ma Trận thưa TF-IDF: {tfidf_matrix.shape} (Tài liệu x Thuộc tính).\n")

    # KHỞI TẠO BƯỚC 2: MÔ HÌNH LSA NGỮ NGHĨA (SVD)
    print("[STEP 2] KHỞI TẠO KHÔNG GIAN PHÂN TÍCH NGỮ NGHĨA ẨN (LSA)")
    print(" -> Đang thi hành kỹ thuật Phân rã Giá trị suy biến (Truncated SVD)...")
    
    n_components = min(300, tfidf_matrix.shape[1] - 1)
    svd_model = TruncatedSVD(n_components=n_components, random_state=42)
    lsa_matrix_raw = svd_model.fit_transform(tfidf_matrix)
    
    # NÂNG CẤP NORMALIZE L2: Ép vector LSA về độ dài 1 (Giúp FAISS Dot Product xử lý như Cosine)
    lsa_matrix = normalize(lsa_matrix_raw, norm='l2', axis=1)

    explained_variance = svd_model.explained_variance_ratio_.sum() * 100
    print(f" -> Kích thước Ma Trận không gian Dense LSA: {lsa_matrix.shape} (Tài liệu x Chủ đề ẩn).")
    print(f" -> Lượng thông tin nguyên bản giữ lại (Explained Variance): {explained_variance:.2f}%\n")

    # BƯỚC I/O LƯU TRỮ VÀO ĐĨA CỨNG
    print("[I/O] LƯU TRỮ CẤU TRÚC HỖ TRỢ VÀO DISK (Dạng .PKL)")
    base_dir = os.path.dirname(TFIDF_MATRIX_PATH)
    os.makedirs(base_dir, exist_ok=True)
    
    paths = {
        "tfidf_matrix": TFIDF_MATRIX_PATH,
        "vectorizer": os.path.join(base_dir, "tfidf_vectorizer.pkl"),
        "svd_model": os.path.join(base_dir, "svd_model.pkl"),
        "lsa_matrix": os.path.join(base_dir, "lsa_matrix.pkl")
    }

    with open(paths["tfidf_matrix"], 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    with open(paths["vectorizer"], 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(paths["svd_model"], 'wb') as f:
        pickle.dump(svd_model, f)
    with open(paths["lsa_matrix"], 'wb') as f:
        pickle.dump(lsa_matrix, f)

    print("[SUCCESS] Xuất toàn bộ Model Vectorizers và Matrices thành công!")
    for key, p in paths.items():
        size_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"   + {key:<14}: {size_mb:>6.2f} MB")
        
    print("="*80 + "\n")
    return True

if __name__ == "__main__":
    build_search_matrix()