"""Mô đun xây dựng FAISS Index để tối ưu hoá tốc độ truy xuất vector LSA."""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# Thêm đường dẫn gốc để Python nhận diện thư mục hệ thống
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import faiss
except ImportError:
    print("LỖI: Chưa cài đặt thư viện truy vấn meta (Faiss).")
    print("Vui lòng chạy lệnh: pip install faiss-cpu")
    exit(1)

from src.utils.config import TFIDF_MATRIX_PATH, FAISS_INDEX_PATH, PROCESSED_DATA_DIR

def build_faiss_index():
    print("=" * 85)
    print("BƯỚC 3: XÂY DỰNG CHỈ MỤC TÌM KIẾM VECTOR (FAISS INVERTED FILE INDEX)")
    print("=" * 85)
    
    base_dir = os.path.dirname(TFIDF_MATRIX_PATH)
    lsa_path = os.path.join(base_dir, "lsa_matrix.pkl")
    data_path = os.path.join(PROCESSED_DATA_DIR, "movies_processed.csv")
    
    if not os.path.exists(lsa_path):
        print(f"[ERROR] Không tìm thấy file nguồn LSA: {lsa_path}.")
        print("Vui lòng chạy file `tfidf_vectorizer.py` trước khi build index.")
        return False
        
    print("[INFO] Đang nạp hệ cơ sở LSA Matrix vào bộ nhớ tạm...")
    with open(lsa_path, 'rb') as f:
        lsa_matrix = pickle.load(f)
        
    # Nạp mapping Index -> movieId
    df = pd.read_csv(data_path, usecols=['movieId'])
        
    # FAISS yêu cầu bắt buộc kiểu dũ liệu ma trận phải là float32
    data_matrix = lsa_matrix.astype(np.float32)
    num_docs, dim = data_matrix.shape
    
    print(f" - Tổng lượng tài liệu tham gia (M): {num_docs:,}")
    print(f" - Số chiều Không gian Vector  (N): {dim}")
    
    print("[INFO] Đang tối ưu hoá không gian đa chiều (L2 Normalization)...")
    # CHUẨN HOÁ VECTOR (Rất quan trọng để DotProduct trở thành Cosine Similarity)
    faiss.normalize_L2(data_matrix)
    
    # =========================================================================
    # NÂNG CẤP LỚN: TỪ BRUTE-FORCE FLAT_IP SANG HỆ THỐNG PHÂN CỤM IVFFLAT
    # =========================================================================
    print("[INFO] Khởi tạo hệ thống phân cụm thuật toán IVFFlat (Inverted File)...")
    
    # Tính toán số lượng cụm (clusters) tối ưu theo công thức: 4 * sqrt(N)
    nlist = int(4 * np.sqrt(num_docs))
    # Đặt ngưỡng thấp nhất là nlist = 50 cho an toàn nếu data nhỏ
    nlist = max(50, nlist) 
    
    # Quản lý khoảng cách Cosine bằng FlatIP 
    quantizer = faiss.IndexFlatIP(dim)
    
    # Khởi tạo Index Phân Cụm
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    print(f" -> Thuật toán K-Means đang huấn luyện {nlist} cụm đại diện (Centroids)...")
    # Train bộ phân cụm với dữ liệu
    index.train(data_matrix)
    
    print("[INFO] Đang phân bổ dữ liệu (Vectors) vào các Cụm tương ứng...")
    index.add(data_matrix)
    
    print(f" -> Hoàn tất nhúng! Tổng số điểm vector tại index: {index.ntotal:,}")
    
    # BƯỚC I/O: LƯU TRỮ CHỈ MỤC VÀ BẢNG MAP ID
    faiss.write_index(index, FAISS_INDEX_PATH)
    
    # Lưu file mapping ID phòng hờ scale dữ liệu cực lớn sau này mất đồng bộ Df_Index
    mapping_path = os.path.join(base_dir, "faiss_movie_map.npy")
    np.save(mapping_path, df['movieId'].values)
    
    size_mb = os.path.getsize(FAISS_INDEX_PATH) / (1024 * 1024)
    print(f"\n[SUCCESS] Đã lưu mô hình chỉ mục hoàn chỉnh tại: {FAISS_INDEX_PATH} ({size_mb:.2f} MB)")
    print(f"[SUCCESS] Đã xuất file Mapping FAISS-TO-ID tại: {mapping_path}")
    print("=" * 85 + "\n")
    
    return True

if __name__ == "__main__":
    build_faiss_index()