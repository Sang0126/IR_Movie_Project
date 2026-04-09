# -*- coding: utf-8 -*-
import os
import sys
import math
import random
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.search.search_engine import MovieSearchEngine
from src.utils.config import PROCESSED_DATA_DIR

random.seed(42)

# ==============================================================================
# BỘ CÔNG CỤ ĐO LƯỜNG IR METRICS (PRECISION, RECALL, HITRATE, NDCG, MRR)
# ==============================================================================
def get_metrics_at_k(retrieved_ids, expected_ids, k=10):
    if not expected_ids:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    top_k_retrieved = retrieved_ids[:k]
    hits = set(top_k_retrieved).intersection(set(expected_ids))
    num_hits = len(hits)

    precision = num_hits / len(top_k_retrieved) if len(top_k_retrieved) > 0 else 0.0
    recall = num_hits / len(expected_ids)
    hit_rate = 1.0 if num_hits > 0 else 0.0

    mrr = 0.0
    for i, item_id in enumerate(top_k_retrieved):
        if item_id in expected_ids:
            mrr = 1.0 / (i + 1)
            break

    dcg = 0.0
    for i, item_id in enumerate(top_k_retrieved):
        if item_id in expected_ids:
            dcg += 1.0 / math.log2(i + 2)
            
    idcg = 0.0
    ideal_hits = min(len(expected_ids), k)
    for i in range(ideal_hits):
        idcg += 1.0 / math.log2(i + 2)

    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return precision, recall, hit_rate, ndcg, mrr

# ==============================================================================
# TẬP DỮ LIỆU TRUY VẤN NGỮ NGHĨA KHÓ (Ép LSA thể hiện sức mạnh)
# ==============================================================================
SEMANTIC_QUERIES_MAP = {
    "billionaire playboy superhero in iron suit": "Iron Man",
    "batman fighting the joker in gotham city": "The Dark Knight",
    "spider man bitten by radioactive spider": "Spider-Man",
    "sinking giant ship jack and rose": "Titanic",
    "space time travel wormhole": "Interstellar",
    "dinosaur theme park disaster": "Jurassic Park",
    "robot finding plant on ruined earth": "WALL·E"
}

def create_heuristic_ground_truth(df, total_queries=200):
    print("Đang tự động sinh tập Ground Truth từ cấu trúc Database...")
    ground_truth = []
    
    # Bỏ sẵn các câu hỏi Ngữ nghĩa (LSA) vào để đánh giá mảng cốt truyện
    if 'title_clean' in df.columns or 'title' in df.columns:
        title_col = 'title_clean' if 'title_clean' in df.columns else 'title'
        for q, expected_title in SEMANTIC_QUERIES_MAP.items():
            expected_ids = df[df[title_col].astype(str).str.lower().str.contains(expected_title.lower(), regex=False)]['movieId'].tolist()
            if expected_ids:
                ground_truth.append({"query": q, "expected_ids": expected_ids, "type": "Semantic/Plot"})

    queries_per_type = (total_queries - len(ground_truth)) // 4

    # Luật Đạo diễn
    if 'director' in df.columns:
        directors = df.dropna(subset=['director'])['director'].unique()
        for _ in range(queries_per_type):
            d = random.choice(directors)
            expected = df[df['director'] == d]['movieId'].tolist()
            ground_truth.append({"query": f"directed by {d}", "expected_ids": expected[:10], "type": "Director"})

    # Luật Diễn viên
    if 'cast_str' in df.columns:
        casts_df = df.dropna(subset=['cast_str'])
        for _ in range(queries_per_type):
            row = casts_df.sample(1).iloc[0]
            actors = str(row['cast_str']).split(', ')
            actor = random.choice(actors[:3]) if actors else "Tom Hanks"
            expected = df[df['cast_str'].str.contains(actor, na=False, case=False, regex=False)]['movieId'].tolist()
            ground_truth.append({"query": f"starring {actor}", "expected_ids": expected[:10], "type": "Actor"})

    # Luật Thể loại 
    if 'genres' in df.columns or 'genres_list' in df.columns:
        genre_col = 'genres' if 'genres' in df.columns else 'genres_list'
        genres_df = df.dropna(subset=[genre_col])
        pop_genres = ["Action", "Comedy", "Sci-Fi", "Romance", "Horror", "Thriller", "Animation"]
        for _ in range(queries_per_type):
            g = random.choice(pop_genres)
            expected = genres_df[genres_df[genre_col].str.contains(g, na=False, case=False, regex=False)]['movieId'].tolist()
            ground_truth.append({"query": f"best {g} movies", "expected_ids": expected[:10], "type": "Genre"})

    # Luật Tên Phim
    if 'title_clean' in df.columns:
        titles_df = df.dropna(subset=['title_clean'])
        titles_df = titles_df[titles_df['title_clean'].astype(str).str.count(' ') >= 2]
        for _ in range(queries_per_type):
            row = titles_df.sample(1).iloc[0]
            words = str(row['title_clean']).split()
            query = " ".join(random.sample(words, 2))
            ground_truth.append({"query": query, "expected_ids": [row['movieId']], "type": "Partial Title"})

    return [q for q in ground_truth if len(q['expected_ids']) > 0]


# ==============================================================================
# HÀM RUN EVALUATION CHÍNH
# ==============================================================================
def run_evaluation():
    data_path = os.path.join(PROCESSED_DATA_DIR, "movies_processed.csv")
    print(f"Nạp Database gốc từ: {data_path}")
    if not os.path.exists(data_path): return

    df = pd.read_csv(data_path, low_memory=False)

    print("Khởi động Search Engine (TF-IDF & FAISS/LSA)...")
    engine = MovieSearchEngine()
    engine.load_index()
    if not engine.is_loaded: return

    test_cases = create_heuristic_ground_truth(df, total_queries=200)
    print(f"Đã tạo thành công {len(test_cases)} Query để test hệ thống.\n")

    scenarios = [
        (1.0, "[MÔ HÌNH 1] TF-IDF THUẦN (Khớp bề mặt)"),
        (0.0, "[MÔ HÌNH 2] LSA THUẦN (Khớp ngữ nghĩa)"),
        (0.5, "[MÔ HÌNH 3] HYBRID LSA+TFIDF (Cân bằng)")
    ]

    report = []
    category_data = []
    TOP_K = 10

    for alpha_val, scenario_name in scenarios:
        print(f"Đang Evaluation: {scenario_name} | Alpha: {alpha_val}")
        
        sum_p = sum_r = sum_hr = sum_ndcg = sum_mrr = sum_latency = 0.0
        n_queries = len(test_cases)
        type_metrics = {}

        for case in test_cases:
            q_type = case['type']
            if q_type not in type_metrics:
                type_metrics[q_type] = {"MRR": 0.0, "HR": 0.0, "Count": 0}

            # Nâng cấp: Theo dõi độ trễ (Latency/Tốc độ của FAISS)
            start_time = time.time()
            results = engine.search_with_reranking(query=case['query'], top_k=TOP_K, alpha=alpha_val)
            end_time = time.time()
            
            sum_latency += (end_time - start_time)

            retrieved_ids = results['movieId'].tolist() if not results.empty else []
            p, r, hr, ndcg, mrr = get_metrics_at_k(retrieved_ids, case['expected_ids'], k=TOP_K)
            
            sum_p += p; sum_r += r; sum_hr += hr; sum_ndcg += ndcg; sum_mrr += mrr
            
            type_metrics[q_type]["MRR"] += mrr
            type_metrics[q_type]["HR"] += hr
            type_metrics[q_type]["Count"] += 1

        avg_latency = (sum_latency / n_queries) * 1000 # Đổi ra milliseconds
        report.append({
            "Mô hình": scenario_name, "Alpha": alpha_val, "Latency_ms": avg_latency,
            "MRR": sum_mrr / n_queries, "Precision": (sum_p / n_queries) * 100,
            "Recall": (sum_r / n_queries) * 100, "HitRate": (sum_hr / n_queries) * 100,
            "NDCG": (sum_ndcg / n_queries) * 100
        })

        for q_type, vals in type_metrics.items():
            if vals["Count"] > 0:
                category_data.append({
                    "Config": scenario_name, "Category": q_type, 
                    "MRR": vals["MRR"]/vals["Count"], "HitRate": (vals["HR"]/vals["Count"])*100
                })

    # ==============================================================================
    # XUẤT BẢNG BÁO CÁO MÀN HÌNH & FILE CSV
    # ==============================================================================
    print("\n" + "="*125)
    print("BẢNG TỔNG QUAN (IR METRICS & LATENCY)".center(125))
    print("="*125)
    print(f"{'Model Configuration':<40} | {'Latency':<8} | {'MRR':<8} | {'Precision':<10} | {'Recall':<10} | {'HitRate':<10} | {'NDCG':<10}")
    print("-" * 125)
    for r in report:
        print(f"{r['Mô hình']:<40} | {r['Latency_ms']:>5.1f} ms | {r['MRR']:<8.4f} | {r['Precision']:>8.2f}% | {r['Recall']:>8.2f}% | {r['HitRate']:>8.2f}% | {r['NDCG']:>8.2f}%")
    print("="*125)

    print("\n" + "="*75)
    print("PHÂN TÍCH NHÓM Ý ĐỊNH TRUY VẤN (INTENT BREAKDOWN)".center(75))
    print("="*75)
    print(f"{'Model Config':<20} | {'Query Type':<15} | {'MRR':<8} | {'HitRate':<10}")
    print("-" * 75)
    for r in category_data:
        if "HYBRID" in r["Config"]:
            print(f"{'HYBRID LSA+TFIDF':<20} | {r['Category']:<15} | {r['MRR']:<8.4f} | {r['HitRate']:>8.2f}%")
    print("="*75)

    # Nâng cấp: Lưu file Evaluation
    report_df = pd.DataFrame(report)
    out_csv = os.path.join(PROCESSED_DATA_DIR, "evaluation_report.csv")
    report_df.to_csv(out_csv, index=False)
    print(f"\n[SUCCESS] Đã kết xuất file báo cáo tại: {out_csv}")

if __name__ == "__main__":
    run_evaluation()