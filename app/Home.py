# -*- coding: utf-8 -*-
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import APP_TITLE

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")

# ==========================================
# CẤU HÌNH THANH SIDEBAR BÊN TRÁI
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3171/3171927.png", width=100)  # Thêm icon phim
    st.markdown("## 🎬 DỰ ÁN TRUY HỒI THÔNG TIN")
    st.markdown("---")
    
    st.markdown("### 📌 **Thông tin Đồ án:**")
    st.markdown("""
    - **Môn học:** Information Retrieval (Truy hồi thông tin)
    - **Kiến trúc đề xuất:** Hybrid Search (TF-IDF + LSA) tích hợp FAISS Re-ranking.
    - **Bộ dữ liệu gốc:** *MovieLens + TMDB* (Hơn 9000 tác phẩm điện ảnh).
    """)
    
    st.markdown("---")
    st.info("💡 **Hướng dẫn:** Chuyển sang trang **System Engine** ở trên để bắt đầu trải nghiệm cỗ máy AI.")

# ==========================================
# GIAO DIỆN CHÍNH Ở GIỮA
# ==========================================
st.markdown("""
<style>
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        background: linear-gradient(to right, #ff416c, #ff4b2b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 2rem;
        margin-bottom: 10px;
    }
    .sub-title { text-align: center; color: #a1a1aa; font-size: 1.3rem; margin-bottom: 50px; letter-spacing: 1px; }
    .stat-container { background: radial-gradient(circle at 10% 20%, rgba(40,40,55,1) 0%, rgba(20,20,30,1) 90%); padding: 2rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.05); }
    .stat-text { color: #8e8ea0; font-size: 1.1rem;}
    .stat-num { font-size: 3.5rem; font-weight: 900; background: linear-gradient(to right, #4CAF50 0%, #38ef7d 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
    .instruction-box { background: rgba(40,40,55,0.5); padding: 20px; border-radius: 8px; border-left: 4px solid #ff4b2b; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">MULTIVERSE MOVIE ENGINE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">He Thong Khai Pha Du Lieu Dien Anh - Su dung Kien Truc AI Ngu Nghia</div>', unsafe_allow_html=True)

# Khối Thống kê đập vào mắt
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown('<div class="stat-container"><h2 class="stat-num">9,499</h2><p class="stat-text">Bo Phim Doc Quyen</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-container"><h2 class="stat-num">10K+</h2><p class="stat-text">Cum Tu Khoa Ngu Nghia</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-container"><h2 class="stat-num">150</h2><p class="stat-text">Chieu Khong Gian LSA</p></div>', unsafe_allow_html=True)

st.markdown("""
<div class="instruction-box">
    <h3 style="color: white; margin-top:0;">Huong Dan Su Dung</h3>
    <p style="color: #d4d4d8; font-size: 1.1rem;">
        Hệ thống được phát triển để xử lý các truy vấn ngôn ngữ tự nhiên phức tạp. Vui lòng bấm vào phần <b>"System Engine"</b> ở thanh công cụ bên trái (Sidebar) để bắt đầu sử dụng Động cơ Tìm kiếm và Gợi ý mô phỏng ngữ nghĩa.
    </p>
</div>
""", unsafe_allow_html=True)