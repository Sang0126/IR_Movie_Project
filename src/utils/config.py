"""File cấu hình dự án."""

import os

# Thư mục gốc của dự án
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Đường dẫn thư mục data
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')

#  Đường dẫn Embeddings 
EMBEDDINGS_DIR = os.path.join(DATA_DIR, 'embeddings')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)
TFIDF_MATRIX_PATH = os.path.join(EMBEDDINGS_DIR, 'tfidf_matrix.pkl')
TFIDF_VECTORIZER_PATH = os.path.join(EMBEDDINGS_DIR, 'tfidf_vectorizer.pkl')

# Đường dẫn dataset MovieLens
MOVIELENS_DIR = os.path.join(RAW_DATA_DIR, 'ml-latest-small')
MOVIELENS_MOVIES = os.path.join(MOVIELENS_DIR, 'movies.csv')
MOVIELENS_RATINGS = os.path.join(MOVIELENS_DIR, 'ratings.csv')
MOVIELENS_TAGS = os.path.join(MOVIELENS_DIR, 'tags.csv')
MOVIELENS_LINKS = os.path.join(MOVIELENS_DIR, 'links.csv')

# Đường dẫn dataset TMDB
TMDB_DIR = os.path.join(RAW_DATA_DIR, 'archive')
TMDB_METADATA = os.path.join(TMDB_DIR, 'movies_metadata.csv')
TMDB_CREDITS = os.path.join(TMDB_DIR, 'credits.csv')
TMDB_KEYWORDS = os.path.join(TMDB_DIR, 'keywords.csv')

# Đường dẫn file sau khi đã xử lý
PROCESSED_MOVIES = os.path.join(PROCESSED_DATA_DIR, 'movies_processed.csv')
PROCESSED_MERGED = os.path.join(PROCESSED_DATA_DIR, 'movies_merged.csv')

# Đường dẫn file embedding
TFIDF_MATRIX = os.path.join(EMBEDDINGS_DIR, 'tfidf_matrix.pkl')
TFIDF_VECTORIZER = os.path.join(EMBEDDINGS_DIR, 'tfidf_vectorizer.pkl')

# FAISS Index Path
FAISS_INDEX_PATH = os.path.join(MODELS_DIR, "lsa_faiss.index")

# Tham số cho TF-IDF
TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.8
TFIDF_NGRAM_RANGE = (1, 2)

# Tham số tìm kiếm
TOP_K_RESULTS = 20
SIMILARITY_THRESHOLD = 0.1

# Tham số gợi ý phim
N_RECOMMENDATIONS = 10

# Cấu hình streamlit
APP_TITLE = "Hệ thống tìm kiếm và gợi ý phim"
APP_ICON = "🎬"


