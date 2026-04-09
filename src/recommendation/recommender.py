"""Mô đun Động cơ Gợi ý Phim (Recommender System) sử dụng Content-Based Filtering Lai siêu cấp tối ưu."""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import re

from src.utils.config import PROCESSED_DATA_DIR, TFIDF_MATRIX_PATH

class MovieRecommender:
    """Hệ thống Gợi ý phim Tương tự nâng cao."""
    
    def __init__(self):
        self.tfidf_matrix = None
        self.lsa_matrix = None
        self.movies_df = None
        self.is_loaded = False

    def load_models(self):
        """Khởi động hệ thống não bộ từ đĩa cứng (Sử dụng chung Matrix với Search Engine)."""
        print("[INFO] Đang nạp Động cơ Gợi ý Phim (Recommender Core)...")
        
        base_dir = os.path.dirname(TFIDF_MATRIX_PATH)
        paths = {
            "tfidf_matrix": TFIDF_MATRIX_PATH,
            "lsa_matrix": os.path.join(base_dir, "lsa_matrix.pkl")
        }
        
        if not os.path.exists(paths["tfidf_matrix"]):
            print(f"[ERROR] Không tìm thấy TFIDF Matrix: {paths['tfidf_matrix']}")
            return False
            
        with open(paths["tfidf_matrix"], 'rb') as f:
            self.tfidf_matrix = pickle.load(f)
            
        if os.path.exists(paths["lsa_matrix"]):
            with open(paths["lsa_matrix"], 'rb') as f:
                self.lsa_matrix = pickle.load(f)
                
        data_path = os.path.join(PROCESSED_DATA_DIR, "movies_processed.csv")
            
        if not os.path.exists(data_path):
            print(f"[ERROR] Chí Mạng: Không tìm thấy Database Dataset tại {data_path}")
            return False
            
        self.movies_df = pd.read_csv(data_path, low_memory=False)
        self.is_loaded = True
        print("[SUCCESS] Động cơ Gợi ý Phim đã nạp thành công!")
        return True

    def calculate_collection_bonus(self, target_title: str, candidate_titles: pd.Series):
        """
        [NÂNG CẤP BÍ QUYẾT 1]: Thưởng Series (Sequel Bonus). 
        So khớp Tiền tố (Prefix) thay vì cắt Token lỏng lẻo để tránh dính "The Lord of War" với "The Lord of the Rings".
        """
        base_target = re.sub(r'[^\w\s]', '', str(target_title).lower()).strip()
        bonus_scores = np.zeros(len(candidate_titles))
        
        # Nếu tên phim quá ngắn (< 4 ký tự) thì không xử lý Series để tránh loạn
        if len(base_target) < 4:
            return bonus_scores
            
        for idx, title in enumerate(candidate_titles):
            cand_title = re.sub(r'[^\w\s]', '', str(title).lower()).strip()
            # Kiểm tra xem tên phim này có bao hàm tên phim kia ở vị trí đầu câu không
            if cand_title != base_target and (cand_title.startswith(base_target) or base_target.startswith(cand_title)):
                bonus_scores[idx] += 0.2  
                
        return bonus_scores

    def calculate_director_cast_bonus(self, target_director: str, target_cast: str, 
                                      dirs_col: pd.Series, casts_col: pd.Series):
        """
        [NÂNG CẤP BÍ QUYẾT 2]: Thưởng điểm cùng ekip.
        Sử dụng split(',') để giữ nguyên cụm tên riêng (Steven Spielberg không bị nhầm với Steven Seagal).
        """
        bonus_scores = np.zeros(len(dirs_col))
        
        # Tách bằng dấu phẩy và làm sạch khoảng trắng thừa
        target_dir_set = set([d.strip().lower() for d in str(target_director).split(',') if d.strip() and d != 'Unknown'])
        target_cast_set = set([c.strip().lower() for c in str(target_cast).split(',')[:5] if c.strip() and c != 'Unknown'])
        
        for idx, (drv, castv) in enumerate(zip(dirs_col, casts_col)):
            cand_dir_set = set([d.strip().lower() for d in str(drv).split(',') if d.strip()])
            if target_dir_set and not target_dir_set.isdisjoint(cand_dir_set):
                bonus_scores[idx] += 0.15 # Cùng đạo diễn
                
            cand_cast_set = set([c.strip().lower() for c in str(castv).split(',')[:5] if c.strip()])
            common_cast = target_cast_set.intersection(cand_cast_set)
            if common_cast:
                # Tính điểm theo số lượng diễn viên chung (0.05 mỗi người, tối đa 0.15)
                bonus_scores[idx] += min(len(common_cast) * 0.05, 0.15)
                
        return bonus_scores

    def recommend(self, movie_id: int, top_k: int = 5, alpha: float = 0.5):
        """Siêu Thuật Toán Gọi ý đã tối ưu (AI Hybrid + Metadata Core + Popularity)."""
        if not self.is_loaded:
            return pd.DataFrame(), ""

        movie_idx_list = self.movies_df.index[self.movies_df['movieId'] == movie_id].tolist()
        if not movie_idx_list:
            return pd.DataFrame(), ""
            
        target_idx = movie_idx_list[0]
        target_movie = self.movies_df.iloc[target_idx]
        target_title = target_movie['title_clean']

        # 1. Tính điểm Mảng AI NLP
        target_tfidf_vector = self.tfidf_matrix[target_idx]
        tfidf_scores = cosine_similarity(target_tfidf_vector, self.tfidf_matrix).flatten()
        
        if self.lsa_matrix is not None:
            target_lsa_vector = self.lsa_matrix[target_idx].reshape(1, -1)
            lsa_scores = cosine_similarity(target_lsa_vector, self.lsa_matrix).flatten()
            final_scores = (alpha * tfidf_scores) + ((1.0 - alpha) * lsa_scores)
        else:
            final_scores = tfidf_scores

        # [NÂNG CẤP BÍ QUYẾT 3]: Chuẩn hóa điểm AI về [0-1] trước khi gộp để không bị bóp méo
        max_ai_score = final_scores.max()
        if max_ai_score > 0:
            final_scores = final_scores / max_ai_score

        # 2. Tính Tác động Cộng thêm (Metadata Booster)
        bonus_collection = self.calculate_collection_bonus(target_title, self.movies_df['title_clean'])
        bonus_crew = self.calculate_director_cast_bonus(
            target_movie.get('director', ''), target_movie.get('cast_str', ''),
            self.movies_df['director'], self.movies_df['cast_str']
        )
        
        # [NÂNG CẤP BÍ QUYẾT 4]: Điểm thưởng độ tín nhiệm (Popularity Tracker)
        max_pop = self.movies_df['popularity_score'].max()
        pop_array = (self.movies_df['popularity_score'].fillna(0.0).values / max_pop) * 0.05
        
        # 3. Tổng hợp Hệ số (Hybrid Blending)
        final_scores = final_scores + bonus_collection + bonus_crew + pop_array

        # Xóa chính phim đang xét khỏi kết quả
        final_scores[target_idx] = -1.0
        
        # Chọn top K ứng viên có Rating cao nhất trong số đó
        top_indices = final_scores.argsort()[-top_k:][::-1]
        
        res_df = self.movies_df.iloc[top_indices].copy()
        res_df['similarity_score'] = final_scores[top_indices]
        
        return res_df, target_title