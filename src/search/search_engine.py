"""Mô đun Động cơ Tìm kiếm (Search Engine) sử dụng Hybrid Search (TF-IDF + FAISS)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
import pickle
import unicodedata
import re
try:
    import faiss
except ImportError:
    pass

from sklearn.metrics.pairwise import cosine_similarity
from src.utils.config import PROCESSED_DATA_DIR, TFIDF_MATRIX_PATH, FAISS_INDEX_PATH

class MovieSearchEngine:
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.svd_model = None
        self.faiss_index = None
        self.movies_df = None
        self.is_loaded = False
    
    def load_index(self):
        print("[INFO] Dang nap Dong co Tim kiem Kep (TF-IDF & FAISS)...")
        base_dir = os.path.dirname(TFIDF_MATRIX_PATH)
        paths = {
            "tfidf_matrix": TFIDF_MATRIX_PATH,
            "vectorizer": os.path.join(base_dir, "tfidf_vectorizer.pkl"),
            "svd_model": os.path.join(base_dir, "svd_model.pkl"),
            "faiss_index": FAISS_INDEX_PATH
        }
        try:
            with open(paths["vectorizer"], 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(paths["tfidf_matrix"], 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            if os.path.exists(paths["faiss_index"]) and os.path.exists(paths["svd_model"]):
                with open(paths["svd_model"], 'rb') as f: 
                    self.svd_model = pickle.load(f)
                self.faiss_index = faiss.read_index(paths["faiss_index"])
                
            data_path = os.path.join(PROCESSED_DATA_DIR, "movies_processed.csv")
            self.movies_df = pd.read_csv(data_path, low_memory=False)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[ERROR] KHONG THE NAP MODEL: {e}")
            return False
    
    def preprocess_query(self, query):
        if not query or not isinstance(query, str): return ""
        query = unicodedata.normalize('NFKD', query).encode('ascii', 'ignore').decode('utf-8')
        query = re.sub(r'[^\w\s]', ' ', query.lower().strip())
        return re.sub(r'\s+', ' ', query)
    
    # [NÂNG CẤP]: Tăng min_similarity lên 0.05 để chặn phim rác hiển thị ra kết quả
    def retrieve_candidates(self, query, top_k=100, min_similarity=0.05, alpha=0.6):
        if not self.is_loaded: return pd.DataFrame()
        cleaned_query = self.preprocess_query(query)
        if not cleaned_query: return pd.DataFrame()
            
        query_tfidf = self.vectorizer.transform([cleaned_query])
        final_scores = np.zeros(len(self.movies_df))
        
        # 1. Quét TF-IDF (Tốc độ Cực Gấp vì dùng Tích vô hướng Sparse)
        tfidf_scores = cosine_similarity(query_tfidf, self.tfidf_matrix).flatten()
        final_scores += (tfidf_scores * alpha)
        
        # 2. Quét FAISS (Chỉ lấy 1000 Vector gần nhất, giải quyết O(N) Tốn Thời Gian)
        if self.faiss_index is not None and self.svd_model is not None:
            query_lsa = self.svd_model.transform(query_tfidf).astype(np.float32)
            faiss.normalize_L2(query_lsa)
            
            # [NÂNG CẤP XỊN]: Khắc phục vòng lặp lấy toàn bộ data, chỉ lấy đủ top ứng viên
            top_faiss_candidates = min(1000, len(self.movies_df))
            D, I = self.faiss_index.search(query_lsa, top_faiss_candidates) 
            
            for score, doc_id in zip(D[0], I[0]):
                if doc_id >= 0 and doc_id < len(self.movies_df) and score > 0: 
                    final_scores[doc_id] += (score * (1.0 - alpha))

        valid_indices = np.where(final_scores >= min_similarity)[0]
        if len(valid_indices) == 0: return pd.DataFrame()
            
        sorted_indices = valid_indices[np.argsort(final_scores[valid_indices])[::-1]]
        top_indices = sorted_indices[:top_k]
        
        results = self.movies_df.iloc[top_indices].copy()
        results['similarity_score'] = final_scores[top_indices]
        return results

    def fetch_top_recommends(self, query, final_k=10, alpha=0.6):
        candidates = self.retrieve_candidates(query, top_k=100, min_similarity=0.05, alpha=alpha)
        if candidates.empty: return candidates
    
        rerank_scores = []
        
        # Lấy giá trị lớn nhất Mảng để Chuẩn hóa (Normalize) về thang [0-1]
        max_pop = candidates['popularity_score'].max() if 'popularity_score' in candidates.columns else 1
        if max_pop == 0 or pd.isna(max_pop): max_pop = 1
            
        max_sim = candidates['similarity_score'].max()
        if max_sim == 0 or pd.isna(max_sim): max_sim = 1.0
        
        for idx, movie in candidates.iterrows():
            sim_score = movie['similarity_score']
            rating = movie.get('avg_rating', 0)
            num_ratings = movie.get('num_ratings', 0)
            popularity = movie.get('popularity_score', 0)
            
            # [NÂNG CẤP RATE]: Đưa toàn bộ chỉ số về khoảng [0, 1] trước khi gộp Hybrid
            sim_norm = float(sim_score) / float(max_sim)
            rating_norm = float(rating) / 5.0 if pd.notna(rating) else 0.5
            trust_factor = min(float(num_ratings) / 100.0, 1.0) if pd.notna(num_ratings) else 0.1
            pop_norm = float(popularity) / max_pop if pd.notna(popularity) else 0.0
        
            # Trọng số chuẩn xác: Ưu tiên Độ khớp (0.7) - Chất lượng phim (0.2) - Độ hot (0.1)
            hybrid_score = (0.7 * sim_norm) + (0.2 * (rating_norm * trust_factor)) + (0.1 * pop_norm)
            rerank_scores.append(hybrid_score)
    
        candidates['hybrid_score'] = rerank_scores
        return candidates.sort_values('hybrid_score', ascending=False).head(final_k)

    def search_with_reranking(self, query, top_k=10, alpha=0.6):
        return self.fetch_top_recommends(query, final_k=top_k, alpha=alpha)