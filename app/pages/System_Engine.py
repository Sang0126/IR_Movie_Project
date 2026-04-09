# -*- coding: utf-8 -*-
import streamlit as st
import sys
import os
import pandas as pd
import json
import urllib.request
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.search.search_engine import MovieSearchEngine
from src.recommendation.recommender import MovieRecommender
from src.utils.config import APP_TITLE

# [NÂNG CẤP BẢO MẬT]: Lấy API Key từ Biến Môi Trường (Mặc định giữ key cũ để bạn dễ chạy test)
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "1e80f9dd12ff7f16ef0ae1f65d64cc7f")

st.set_page_config(page_title=f"Core Engine - {APP_TITLE}", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# KHỞI TẠO SESSION STATE VÀ CALLBACKS
# ==========================================
if "search_results" not in st.session_state: st.session_state.search_results = None
if "search_query" not in st.session_state: st.session_state.search_query = ""
if "q_input" not in st.session_state: st.session_state.q_input = ""
if "active_recommendation" not in st.session_state: st.session_state.active_recommendation = {} 
if "active_trailers" not in st.session_state: st.session_state.active_trailers = {}
if "top_k_search" not in st.session_state: st.session_state.top_k_search = 10
if "search_history" not in st.session_state: st.session_state.search_history = [] 
if "trigger_search" not in st.session_state: st.session_state.trigger_search = False

def run_search_action(kw=None):
    if kw: 
        st.session_state.search_query = kw
        st.session_state.q_input = kw
    else:
        st.session_state.search_query = st.session_state.q_input
        
    st.session_state.trigger_search = True
    st.session_state.active_recommendation = {}
    st.session_state.active_trailers = {}

def show_trailer_action(movie_id):
    st.session_state.active_trailers[movie_id] = True

def load_recommendation_action(movie_id, title):
    k_val = st.session_state[f"sel_rec_{movie_id}"]
    st.session_state.active_recommendation[f"rec_{movie_id}"] = {"k": k_val, "title": title}

def clear_system_cache():
    st.session_state.search_history = []
    st.session_state.search_results = None
    st.session_state.active_recommendation = {}
    st.session_state.active_trailers = {}
    st.session_state.search_query = ""
    st.session_state.q_input = ""

# ==========================================
# TIỆN ÍCH UI NÂNG CAO (HIGHLIGHT & EXPLAIN)
# ==========================================
def highlight_keyword(text, query):
    """Làm nổi bật các từ khóa tìm kiếm trong phần nội dung."""
    if not query or str(text) == 'N/A': return text
    terms = [t for t in re.split(r'\W+', query.lower()) if len(t) > 2]
    highlighted = str(text)
    for term in terms:
        # Tô vàng các từ khóa dài hơn 2 ký tự trùng với câu tìm kiếm
        highlighted = re.sub(
            f'(?i)({re.escape(term)})', 
            r'<mark style="background-color: rgba(255,235,59,0.4); color: #fff; padding: 0 3px; border-radius: 3px;">\1</mark>', 
            highlighted
        )
    return highlighted

def explain_recommendation(target_dir, target_cast, cand_dir, cand_cast):
    """Trích xuất lý do vì sao Hệ thống Gợi ý phim này (Explainable AI)."""
    explanations = []
    
    # Check Đạo diễn khối
    if cand_dir != 'N/A' and cand_dir != 'Unknown' and any(d in target_dir for d in cand_dir.split(',')):
        explanations.append("Cùng Đạo Diễn")
        
    # Check Diễn viên khối
    t_cast_set = set([c.strip().lower() for c in target_cast.split(',')])
    c_cast_set = set([c.strip().lower() for c in cand_cast.split(',')])
    if t_cast_set.intersection(c_cast_set):
        explanations.append("Chung Diễn Viên")
        
    if not explanations:
        explanations.append("Tương đồng Chủ đề")
        
    html = " ".join([f"<span class='badge-explain'>{ex}</span>" for ex in explanations])
    return f"<div style='margin-bottom: 8px;'>{html}</div>"

# ==========================================
# CẤU HÌNH THANH SIDEBAR BÊN TRÁI
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3171/3171927.png", width=70)
    st.markdown("### ⚙️ **BẢNG ĐIỀU KHIỂN CHUYÊN LÝ**")
    st.markdown("Cấu hình trọng số thuật toán tìm kiếm và đề xuất.")
    
    st.markdown("---")
    st.markdown("#### Thiết lập Tìm kiếm (Search)")
    sys_alpha_search = st.slider("Tỉ lệ nội suy TF-IDF vs FAISS:", 0.0, 1.0, 0.6, 0.1, help="Định mức ưu tiên: Trọng số 0.6 = 60% Keyword/TF-IDF và 40% Ngữ nghĩa/LSA.")
    
    st.markdown("---")
    st.markdown("#### Thiết lập Đề xuất (Recommender)")
    sys_alpha_rec = st.slider("Trọng số không gian Recommender:", 0.0, 1.0, 0.5, 0.1, help="Trọng số cân bằng mức độ phổ biến (Popularity) và độ tương đồng văn bản.")
    
    st.markdown("---")
    st.button("🗑️ Xóa Cache Bộ nhớ", use_container_width=True, on_click=clear_system_cache)

# CSS TUỲ CHỈNH CẤP CAO
st.markdown("""
<style>
    .hero-header { font-size: 2.5rem; font-weight: 800; text-align: center; margin-top: 10px; margin-bottom: 5px; background: linear-gradient(to right, #4CAF50, #8bc34a); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-subtitle { text-align: center; color: #a1a1aa; font-size: 1.15rem; margin-bottom: 20px; }
    .suggestion-box { margin-bottom: 25px; background: rgba(30, 30, 46, 0.4); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);}
    .sug-title { color: #8e8ea0; font-size: 0.95rem; font-weight: bold; margin-bottom: 10px; display: block;}
    .movie-card { background: rgba(30, 30, 46, 0.6); border-radius: 12px; padding: 22px; margin-bottom: 20px; border-left: 5px solid #0078D7; box-shadow: 0 4px 15px rgba(0,0,0,0.2); transition: all 0.3s ease; }
    .movie-card:hover { transform: translateY(-3px); box-shadow: 0 6px 20px rgba(0, 120, 215, 0.2); background: rgba(40, 40, 56, 0.9);}
    .title-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 5px; }
    .m-title { font-size: 1.5rem; font-weight: 700; color: #ffffff; margin: 0; line-height: 1.3;}
    .m-genres { font-size: 0.95rem; color: #a1a1aa; margin-bottom: 10px; font-style: italic;}
    .m-match-search { background: #0078D7; color: white; padding: 5px 12px; border-radius: 20px; font-size: 0.95rem; font-weight: bold; white-space: nowrap;}
    .streamlit-expanderHeader { background-color: transparent !important; color: #fff; font-size: 1.05rem; font-weight: bold;}
    .meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; margin-bottom: 15px; background: rgba(0,0,0,0.3); padding: 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); }
    .m-item strong { color: #0078D7; display: block; margin-bottom: 3px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px;}
    .m-desc { color: #cbd5e1; font-size: 1.05rem; line-height: 1.6; padding: 18px; background: rgba(0,0,0,0.2); border-radius: 8px; margin-top: 10px; }
    .nested-rec-box { background: rgba(20,20,30,0.8); border: 1px solid #333; border-left: 4px solid #4CAF50; padding: 20px; border-radius: 8px; margin-bottom: 15px; margin-top: 15px; transition: transform 0.2s; }
    .nested-rec-box:hover { transform: scale(1.01); border-color: #4CAF50; box-shadow: 0 4px 15px rgba(76, 175, 80, 0.15);}
    .rec-title { font-size: 1.3rem; font-weight: 700; color: #ffffff; margin: 0; line-height: 1.3;}
    .m-match-rec { background: #4CAF50; color: white; padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; white-space: nowrap;}
    .badge-explain { display: inline-block; background: #6366f1; color: white; padding: 3px 10px; font-size: 0.8rem; font-weight: 600; border-radius: 12px; border: 1px solid #4f46e5; margin-right: 5px; }
    .trailer-placeholder { background-color: #000; color: white; padding: 20px; text-align:center; border-radius: 8px; margin-top: 10px; border: 1px dashed #555;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_systems():
    s_engine = MovieSearchEngine()
    s_engine.load_index()
    r_engine = MovieRecommender()
    r_engine.load_models()
    return s_engine, r_engine

with st.spinner("Đang khởi tạo hệ thống truy hồi thông tin..."):
    search_engine, rec_engine = load_systems()

if not search_engine.is_loaded or not rec_engine.is_loaded:
    st.error("Lỗi giao tiếp với Model Vector. Vui lòng kiểm tra lại Data Pipeline.")
    st.stop()

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_movie_poster(tmdb_id, title, year):
    safe_title = str(title).encode('ascii', 'ignore').decode().replace(' ', '+') if title else f"ID_{tmdb_id}"
    safe_year = f"+({year})" if pd.notna(year) and str(year) != 'N/A' else ""
    dummy_img = f"https://dummyimage.com/500x750/1a1a2e/4CAF50&text={safe_title}{safe_year}"
    
    if pd.isna(tmdb_id) or str(tmdb_id).strip() == 'nan':
        return dummy_img
        
    try:
        url = f"https://api.themoviedb.org/3/movie/{int(float(tmdb_id))}?api_key={TMDB_API_KEY}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req, timeout=2)
        data = json.loads(response.read().decode())
        
        poster_path = data.get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except Exception:
        pass
    return dummy_img

@st.cache_data(ttl=3600)  
def get_youtube_video_id(title, year):
    try:
        safe_title = str(title).replace(' ', '+')
        safe_year = str(year) if str(year) not in ['N/A', 'nan', ''] else ''
        search_query = f"{safe_title}+trailer+{safe_year}"
        
        # Nâng cấp Regex hạn chế lỗi cấu trúc Web
        req = urllib.request.Request(
            f"https://www.youtube.com/results?search_query={search_query}", 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        html = urllib.request.urlopen(req, timeout=3).read().decode()
        video_ids = re.findall(r"watch\?v=([a-zA-Z0-9_-]{11})", html)
        
        if video_ids:
            return f"https://www.youtube.com/watch?v={video_ids[0]}" 
        return None
    except Exception:
        return None

def format_movie_data(m_data, max_score, base=75.0, top=24.0, is_search=True):
    raw_dir = str(m_data.get('director', 'N/A'))
    director = raw_dir.strip()
    
    raw_cast = str(m_data.get('cast_str', 'N/A'))
    cast_list = [name.strip() for name in raw_cast.split(',') if name.strip()]
    cast = ", ".join(cast_list) if cast_list else "Unknown"
    
    title_col = 'title_clean' if 'title_clean' in m_data else ('original_title' if 'original_title' in m_data else 'title')
    title = m_data.get(title_col, 'Unknown')
    
    raw_year = m_data.get('release_year', 'N/A')
    year = int(raw_year) if pd.notna(raw_year) and str(raw_year).replace('.', '').isdigit() else 'N/A'
    
    genres = str(m_data.get('genres_str', 'N/A')).replace('|', ', ')
    rating = round(m_data.get('avg_rating', 0.0), 1)
    votes = int(m_data.get('num_ratings', 0))
    runtime = int(m_data.get('runtime', 0))
    overview = str(m_data.get('overview', 'N/A'))
    
    score_key = 'hybrid_score' if is_search else 'similarity_score'
    raw_sc = m_data.get(score_key, 0)
    score = base + ((raw_sc / max_score) * top) if max_score > 0 else base
    return title, year, director, cast, genres, rating, votes, runtime, overview, score

# ==========================================
# TRANG SINGLE PAGE: GIAO DIỆN
# ==========================================
st.markdown('<div class="hero-header">HỆ THỐNG TRUY HỒI ĐIỆN ẢNH</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Kiến trúc Tìm kiếm Ngữ nghĩa & Động cơ Phân tích Tương đồng</div>', unsafe_allow_html=True)

col_hist, col_trend = st.columns([1, 1])

with col_hist:
    st.markdown('<div class="suggestion-box"><span class="sug-title">🕒 Lịch sử tham vấn (History):</span>', unsafe_allow_html=True)
    if len(st.session_state.search_history) > 0:
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for i, hist_kw in enumerate(reversed(st.session_state.search_history[-3:])):
            with cols[i % 3]:
                st.button(f"🕒 {hist_kw}", key=f"hist_{i}", use_container_width=True, on_click=run_search_action, args=(hist_kw,))
    else:
        st.markdown("<p style='color: gray; font-style: italic;'>Hệ thống ghi nhận bộ nhớ trống.</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_trend:
    st.markdown('<div class="suggestion-box"><span class="sug-title">⭐ Mẫu truy vấn tiêu biểu (Trending):</span>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.button("🔥 Dinosaur disaster", use_container_width=True, on_click=run_search_action, args=("dinosaur theme park disaster",))
    with c2:
        st.button("🔥 Space survival", use_container_width=True, on_click=run_search_action, args=("man stranded alone on mars",))
    with c3:
        st.button("🔥 Superhero Iron", use_container_width=True, on_click=run_search_action, args=("billionaire playboy superhero in iron suit",))
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
col_input, col_num, col_btn = st.columns([6, 2, 2])
with col_input:
    st.text_input("Nhập dữ kiện kịch bản, tựa đề, nội dung thẻ hoặc tên diễn viên (En):", key="q_input")
with col_num:
    st.selectbox("Số lượng kết quả xuất ra:", options=[5, 10, 20, 50], index=1, key="top_k_search")
with col_btn:
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    st.button("Truy Xuất Dữ Liệu", type="primary", use_container_width=True, on_click=run_search_action)

st.markdown("---")

if st.session_state.trigger_search and st.session_state.search_query:
    active_q = st.session_state.search_query
    if active_q not in st.session_state.search_history:
        st.session_state.search_history.append(active_q)
        
    with st.spinner("Hệ thống đang nội suy thông tin văn bản & vector..."):
        st.session_state.search_results = search_engine.search_with_reranking(
            query=active_q, 
            top_k=st.session_state.top_k_search,
            alpha=sys_alpha_search
        )
    st.session_state.trigger_search = False

if st.session_state.search_results is not None and not st.session_state.search_results.empty:
    results = st.session_state.search_results
    st.success(f"✔️ Truy hồi thông tin hoàn tất! Khai xuất {len(results)} bản ghi thích hợp.")
    
    max_score = results['hybrid_score'].max() if not results.empty else 1.0
    active_q = st.session_state.search_query
    
    for idx, (_, movie) in enumerate(results.iterrows(), 1):
        title, year, director, cast, genres, rating, votes, runtime, overview, score = format_movie_data(movie, max_score, 75.0, 23.0, True)
        movie_id = movie['movieId']
        
        poster_url = fetch_movie_poster(movie.get('tmdbId'), title, year)
        
        # Gọi Hàm Highlight để phát sáng từ khóa trong Overview
        hl_overview = highlight_keyword(overview, active_q)
        
        st.markdown(f"""
        <div class="movie-card">
            <div class="title-row">
                <h3 class="m-title">[{idx}] {title} ({year})</h3>
                <div class="m-match-search">Mức độ tương quan (Relevance): {score:.1f}%</div>
            </div>
            <div class="m-genres">{genres}</div>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander(f"📖 Tra cứu thông tin tác phẩm '{title}'"):
            col_image, col_info = st.columns([1, 3])
            
            with col_image:
                st.image(poster_url, use_container_width=True)
                
            with col_info:
                st.markdown(f"""
                <div class="meta-grid" style="margin-top: 0;">
                    <div class="m-item"><strong>Nhà Chỉ Đạo</strong> {director}</div>
                    <div class="m-item"><strong>Chuẩn Thời Gian</strong> {runtime} Min</div>
                    <div class="m-item"><strong>Thước Đo Đại Chúng</strong> ⭐ {rating}/5.0 (Tham chiếu từ {votes:,} vote)</div>
                    <div class="m-item" style="grid-column: 1 / -1;"><strong>Thông tin diễn viên</strong> {cast}</div>
                </div>
                <div class="m-desc"><strong>Trích lục nội dung:</strong><br>{hl_overview}</div>
                """, unsafe_allow_html=True)
                
                st.markdown("<p style='color:#0078D7; font-weight:bold; margin-top:20px; font-size:1.1rem;'>▶ THỊ CHUẨN MEDIA TỪ YOUTUBE:</p>", unsafe_allow_html=True)
                st.button(f"📺 Biến dịch kết nối Video (Trailer)", key=f"fetch_vid_{movie_id}", on_click=show_trailer_action, args=(movie_id,))
                
                if st.session_state.active_trailers.get(movie_id):
                    with st.spinner("Đang tìm chuỗi nguồn phát đa phương tiện..."):
                        trailer_link = get_youtube_video_id(title, year)
                        if trailer_link: st.video(trailer_link)
                        else: st.markdown("<div class='trailer-placeholder'>Lỗi: Tài nguyên không hợp lệ hoặc bị gián đoạn truyền tải.</div>", unsafe_allow_html=True)
            
            st.markdown("<hr style='border: 1px dashed #444; margin: 20px 0;'>", unsafe_allow_html=True)
            
            cc1, cc2 = st.columns([1, 4])
            with cc1:
                st.selectbox("Số lượng xuất khuyên dùng:", [3, 5, 10], key=f"sel_rec_{movie_id}", label_visibility="collapsed")
            with cc2:
                st.button(f"✨ Kích hoạt Bộ lọc Đề xuất (Recommendation Module)", key=f"btn_rec_{movie_id}", use_container_width=True, on_click=load_recommendation_action, args=(movie_id, title))
            
            if st.session_state.active_recommendation.get(f"rec_{movie_id}"):
                stored_k = st.session_state.active_recommendation[f"rec_{movie_id}"]["k"]
                
                with st.spinner("Khởi động LSA Vector và xử lý độ đo cục bộ..."):
                    nested_rec, _ = rec_engine.recommend(
                        movie_id=movie_id, 
                        top_k=stored_k, 
                        alpha=sys_alpha_rec
                    )
                    
                    if nested_rec is not None and not nested_rec.empty:
                        max_nested_sc = nested_rec['similarity_score'].max()
                        st.markdown(f"<p style='color: #4CAF50; font-size: 1.15rem; font-weight: bold; margin-top: 25px;'>BẢNG GIÁ HOÁ KẾT QUẢ ĐỀ XUẤT TƯƠNG ĐỒNG:</p>", unsafe_allow_html=True)
                        
                        for n_idx, (_, n_movie) in enumerate(nested_rec.iterrows(), 1):
                            n_title, n_year, n_dir, n_cast, n_gen, n_rate, _, _, n_overview, n_score = format_movie_data(n_movie, max_nested_sc, 80.0, 19.0, False)
                            short_ov = (n_overview[:150] + '...') if len(n_overview) > 150 else n_overview
                            n_poster = fetch_movie_poster(n_movie.get('tmdbId'), n_title, n_year)
                            
                            # Cung cấp Lý giải (Explainable AI)
                            explanation_ui = explain_recommendation(director, cast, n_dir, n_cast)

                            n_c1, n_c2 = st.columns([1, 6])
                            with n_c1:
                                st.image(n_poster, use_container_width=True)
                            with n_c2:
                                st.markdown(f"""
                                <div class="nested-rec-box" style="margin-top: 0; padding: 10px;">
                                    <div class="title-row" style="margin-bottom:0px;">
                                        <h4 class="rec-title">{n_idx}. <a href="https://www.youtube.com/results?search_query={str(n_title).replace(' ','+')}+trailer" target="_blank" style="color:white; text-decoration:none;">{n_title} ({n_year})</a></h4>
                                        <div class="m-match-rec">Độ tự tin: {n_score:.1f}%</div>
                                    </div>
                                    {explanation_ui}
                                    <p style="color: #94a3b8; font-size: 0.95rem; margin-top: 5px; margin-bottom: 2px;"><strong>Cầm trịch:</strong> {n_dir} | <strong>Đính giá:</strong> ⭐{n_rate}/5.0</p>
                                    <p style="color: #4CAF50; font-size: 0.9rem; font-style: italic; margin-top: 0px;">{n_gen}</p>
                                    <p style="color: #cbd5e1; font-size: 1rem; margin-top: 10px; margin-bottom:0;">{short_ov}</p>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ Sự cố logic: Chưa thể trích đoạn thông tin bổ trợ.")
elif st.session_state.trigger_search:
    st.warning("⚠️ Từ chối truy cập: Lệnh của bạn không mang lại kết xuất nội dung tương đồng.")