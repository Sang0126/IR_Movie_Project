"""Module tải và gộp dữ liệu từ nhiều nguồn."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
from typing import Optional

# Import tường minh để tránh xung đột biến môi trường
from src.utils.config import (
    PROCESSED_DATA_DIR, 
    MOVIELENS_MOVIES, MOVIELENS_RATINGS, MOVIELENS_TAGS, MOVIELENS_LINKS, 
    TMDB_METADATA, TMDB_CREDITS, TMDB_KEYWORDS, 
    PROCESSED_MERGED
)


class DataLoader:
    """Lớp thực hiện việc tải và gộp dữ liệu từ MovieLens và TMDB."""
    
    def __init__(self) -> None:
        """Khởi tạo DataLoader và các biến lưu trữ dữ liệu."""
        self.movielens_movies: Optional[pd.DataFrame] = None
        self.movielens_ratings: Optional[pd.DataFrame] = None
        self.movielens_tags: Optional[pd.DataFrame] = None
        self.movielens_links: Optional[pd.DataFrame] = None
        
        self.tmdb_metadata: Optional[pd.DataFrame] = None
        self.tmdb_credits: Optional[pd.DataFrame] = None
        self.tmdb_keywords: Optional[pd.DataFrame] = None
        
        self.merged_data: Optional[pd.DataFrame] = None
        
        # Tạo thư mục để lưu các file dữ liệu trung gian trong quá trình gộp
        self.intermediate_dir = os.path.join(PROCESSED_DATA_DIR, 'intermediate_steps')
        os.makedirs(self.intermediate_dir, exist_ok=True)
    
    def _save_intermediate_step(self, df: pd.DataFrame, step_name: str, description: str = "") -> None:
        """Lưu file trung gian sau mỗi bước tiến hành gộp dữ liệu."""
        filepath = os.path.join(self.intermediate_dir, f"{step_name}.csv")
        df.to_csv(filepath, index=False)
        
        file_size = os.path.getsize(filepath) / (1024 * 1024)
        print(f"    + Đã lưu file: {step_name}.csv ({file_size:.2f} MB)")
        
        if description:
            print(f"      Chi tiết: {description}")
    
    def load_movielens(self) -> pd.DataFrame:
        """Tải dữ liệu từ MovieLens."""
        print("Đang tải dữ liệu gốc từ MovieLens...")
        
        self.movielens_movies = pd.read_csv(MOVIELENS_MOVIES)
        print(f" - Tải thành công {len(self.movielens_movies):,} bộ phim.")
        
        self.movielens_ratings = pd.read_csv(MOVIELENS_RATINGS)
        print(f" - Tải thành công {len(self.movielens_ratings):,} lượt đánh giá.")
        
        self.movielens_tags = pd.read_csv(MOVIELENS_TAGS)
        print(f" - Tải thành công {len(self.movielens_tags):,} thẻ người dùng.")
        
        self.movielens_links = pd.read_csv(MOVIELENS_LINKS)
        print(f" - Tải thành công {len(self.movielens_links):,} liên kết ID.")
        
        return self.movielens_movies
    
    def load_tmdb(self) -> pd.DataFrame:
        """Tải dữ liệu từ TMDB và xử lý sơ bộ các ID bị trùng lặp."""
        print("\nĐang tải dữ liệu gốc từ TMDB...")
        
        try:
            self.tmdb_metadata = pd.read_csv(TMDB_METADATA, low_memory=False, on_bad_lines='skip')
            # Loại bỏ các phim bị trùng lặp ID trong bộ gốc của TMDB
            self.tmdb_metadata = self.tmdb_metadata.drop_duplicates(subset=['id'], keep='first')
            self.tmdb_metadata['id'] = self.tmdb_metadata['id'].astype(str)
            print(f" - Tải thành công {len(self.tmdb_metadata):,} bộ phim từ TMDB (đã lọc trùng lặp).")
        except Exception as e:
            print(f"Lỗi khi tải TMDB metadata: {e}")
        
        if os.path.exists(TMDB_CREDITS):
            try:
                self.tmdb_credits = pd.read_csv(TMDB_CREDITS)
                self.tmdb_credits = self.tmdb_credits.drop_duplicates(subset=['id'], keep='first')
                self.tmdb_credits['id'] = self.tmdb_credits['id'].astype(str)
                print(f" - Tải thành công {len(self.tmdb_credits):,} thông tin đội ngũ làm phim.")
            except Exception as e:
                print(f"Lỗi khi tải dữ liệu đội ngũ làm phim: {e}")
        
        if os.path.exists(TMDB_KEYWORDS):
            try:
                self.tmdb_keywords = pd.read_csv(TMDB_KEYWORDS)
                self.tmdb_keywords = self.tmdb_keywords.drop_duplicates(subset=['id'], keep='first')
                self.tmdb_keywords['id'] = self.tmdb_keywords['id'].astype(str)
                print(f" - Tải thành công {len(self.tmdb_keywords):,} danh sách từ khóa.")
            except Exception as e:
                print(f"Lỗi khi tải dữ liệu từ khóa: {e}")
        
        return self.tmdb_metadata

    # =========================================================================
    # CÁC HÀM HỖ TRỢ GỘP DỮ LIỆU (Được tách nhỏ để tối ưu kiến trúc)
    # =========================================================================

    def _prepare_base_movies(self) -> pd.DataFrame:
        """Bước 1 & 2: Chuẩn bị bộ khung ID gốc từ MovieLens."""
        print("\n[BƯỚC 1 & 2] Lấy khung MovieLens và chuẩn hóa định dạng ID liên kết...")
        
        df = self.movielens_movies.merge(self.movielens_links, on='movieId', how='left')
        
        # Lọc bỏ và chuẩn hóa tmdbId
        df = df.dropna(subset=['tmdbId'])
        df['tmdbId'] = df['tmdbId'].astype(int).astype(str)
        
        print(f"  - Số lượng khung phim hợp lệ giữ lại: {len(df):,} phim.")
        return df

    def _merge_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bước 3: Gộp với siêu dữ liệu TMDB."""
        print("\n[BƯỚC 3] Nối với thư viện siêu dữ liệu (metadata) của TMDB...")
        
        df = df.merge(
            self.tmdb_metadata,
            left_on='tmdbId', right_on='id',
            how='left', suffixes=('_ml', '_tmdb')
        )
        
        valid_cnt = df['overview'].notna().sum()
        print(f"  - Có nội dung cốt truyện: {valid_cnt:,} phim ({(valid_cnt/len(df)*100):.1f}%).")
        return df

    def _merge_credits_and_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bước 4 & 5: Gộp dữ liệu đội ngũ làm phim và từ khóa."""
        print("\n[BƯỚC 4 & 5] Bổ sung danh sách diễn viên, đạo diễn và từ khóa...")
        
        if self.tmdb_credits is not None:
            df = df.merge(
                self.tmdb_credits[['id', 'cast', 'crew']], 
                left_on='tmdbId', right_on='id', how='left'
            )
            df = df.loc[:, ~df.columns.duplicated()] # Loại bỏ các cột 'id' rác bị chép đè
            
        if self.tmdb_keywords is not None:
            df = df.merge(
                self.tmdb_keywords[['id', 'keywords']], 
                left_on='tmdbId', right_on='id', how='left'
            )
            df = df.loc[:, ~df.columns.duplicated()]
            
        print(f"  - Cấu trúc nối dữ liệu phụ hoàn tất.")
        return df

    def _merge_ratings_and_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bước 6 & 7: Gộp điểm đánh giá và thẻ gắn chuẩn hóa."""
        print("\n[BƯỚC 6 & 7] Tính điểm đánh giá trung bình và chuẩn hóa User Tags...")
        
        # Tính trung bình đánh giá
        ratings_agg = self.movielens_ratings.groupby('movieId').agg(
            avg_rating=('rating', 'mean'),
            num_ratings=('rating', 'count')
        ).reset_index()
        df = df.merge(ratings_agg, on='movieId', how='left')
        
        # Nhóm Tag: Chuyển chữ thường, Xóa trùng lặp, Nối chuỗi
        if len(self.movielens_tags) > 0:
            tags_agg = self.movielens_tags.groupby('movieId')['tag'].apply(
                lambda x: ' '.join(sorted(set(str(t).lower() for t in x if pd.notna(t))))
            ).reset_index()
            tags_agg.columns = ['movieId', 'user_tags']
            df = df.merge(tags_agg, on='movieId', how='left')
            
        print(f"  - Đánh giá trung bình hệ thống: {df['avg_rating'].mean():.2f}/5.0")
        return df

    # =========================================================================
    # HÀM ĐIỀU PHỐI CHÍNH
    # =========================================================================

    def merge_datasets(self, save_intermediate: bool = True) -> pd.DataFrame:
        """Điều phối toàn bộ quá trình gộp dữ liệu."""
        print("\n" + "="*80)
        print("QUY TRÌNH GỘP DỮ LIỆU (CÓ LƯU FILE TRUNG GIAN)")
        print("="*80)
        
        if self.movielens_movies is None or self.tmdb_metadata is None:
            self.load_movielens()
            self.load_tmdb()
            
        # Chạy pipeline nối dữ liệu
        merged = self._prepare_base_movies()
        if save_intermediate: self._save_intermediate_step(merged, "step1_2_base_links_filtered", "Cấu trúc khung ID gốc chuẩn hóa.")

        merged = self._merge_metadata(merged)
        if save_intermediate: self._save_intermediate_step(merged, "step3_metadata_added", "Đã nối metadata.")

        merged = self._merge_credits_and_keywords(merged)
        if save_intermediate: self._save_intermediate_step(merged, "step4_5_credits_keywords", "Đã nối diễn viên và từ khóa.")

        merged = self._merge_ratings_and_tags(merged)
        if save_intermediate: self._save_intermediate_step(merged, "step6_7_ratings_tags", "Đã nối số liệu đánh giá và thẻ gắn chuẩn hóa.")
        
        self.merged_data = merged
        
        print("\n" + "="*80)
        print("TỔNG KẾT QUÁ TRÌNH GỘP DỮ LIỆU")
        print("="*80)
        print(f"- Số lượng phim hoàn chỉnh: {len(merged):,}")
        print(f"- Số lượng thuộc tính (cột): {len(merged.columns)}")
        print(f"\nCác file thực thi trung gian đã được tự động lưu trữ tại:")
        print(f"  {self.intermediate_dir}")
        print("="*80)
        
        return merged
    
    def get_dataset_info(self) -> None:
        """In bảng tóm tắt nhanh về tình trạng hiện tại của dữ liệu."""
        if self.merged_data is None:
            print("Lỗi: Dữ liệu chưa tồn tại. Vui lòng chạy hàm merge_datasets() trước.")
            return
        
        print("\n" + "="*60)
        print("BẢNG TÓM TẮT DỮ LIỆU SAU KHI GỘP")
        print("="*60)
        print(f"Tổng số bộ phim              : {len(self.merged_data):,}")
        print(f"Số phim có tóm tắt cốt truyện: {self.merged_data['overview'].notna().sum():,}")
        print(f"Tổng số lượng cột            : {len(self.merged_data.columns)}")
        print("="*60)
    
    def save_processed_data(self, filepath: str = PROCESSED_MERGED) -> None:
        """Hành động ghi xuất dữ liệu đã gộp ra một file CSV vật lý trên máy tính."""
        if self.merged_data is None:
            return
        
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        print(f"\nĐang ghi xuất kết quả hoàn chỉnh vào: {filepath} ...")
        self.merged_data.to_csv(filepath, index=False)
        print("Thành công! Khởi tạo file nguyên liệu hoàn tất.")
    
    def load_processed_data(self, filepath: str = PROCESSED_MERGED) -> Optional[pd.DataFrame]:
        """Đọc thẳng dữ liệu đã gộp lúc trước để bỏ qua khâu xử lý lâu lắc."""
        if os.path.exists(filepath):
            print(f"Đang đọc dữ liệu trực tiếp từ: {filepath}...")
            self.merged_data = pd.read_csv(filepath, low_memory=False)
            print(f" - Phục hồi thành công {len(self.merged_data):,} bản ghi.")
            return self.merged_data
        return None
    
    def list_intermediate_files(self) -> None:
        """Danh sách kiểm tra lại toàn bộ file quy trình đã xuất."""
        if not os.path.exists(self.intermediate_dir):
            return
        
        files = [f for f in os.listdir(self.intermediate_dir) if f.endswith('.csv')]
        print("\n" + "="*80)
        print("THỐNG KÊ CÁC TỆP TIN TRUNG GIAN TRONG QUÁ TRÌNH XỬ LÝ")
        print("="*80)
        for file_name in sorted(files):
            size = os.path.getsize(os.path.join(self.intermediate_dir, file_name)) / (1024 * 1024)
            print(f" - Tên file: {file_name:35s} | Kích thước: {size:.2f} MB")
        print("="*80)


if __name__ == "__main__":
    loader = DataLoader()
    loader.merge_datasets(save_intermediate=True)
    loader.get_dataset_info()
    loader.save_processed_data()
    loader.list_intermediate_files()