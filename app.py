"""
=============================================================
  DETEKSI KEMIRIPAN WAJAH — PCA / SVD  (Streamlit App)
  UI: Sleek Black, Navbar atas, Parameter di Sidebar Terbuka
=============================================================
"""

import os
import tempfile
import warnings

warnings.filterwarnings("ignore")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA as SklearnPCA

# ──────────────────────────────────────────────
#  HALAMAN CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FaceVec — Deteksi Kemiripan Wajah",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded", # Otomatis terbuka di awal, bisa ditutup-buka secara fleksibel
)

# ──────────────────────────────────────────────
#  CSS — Sleek Black, Kepadatan Tinggi & Padding 5%
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main {
    background-color: #080808 !important;
    color: #d4d4d4 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Mengembalikan & Menghias Tombol Pemicu Sidebar ── */
[data-testid="stHeader"] {
    background-color: transparent !important;
    z-index: 999 !important;
}
[data-testid="collapsedControl"] {
    color: #6366f1 !important; /* Warna tombol panah dibuat ungu mencolok */
    background: #0f0f1a !important;
    border: 1px solid #262626 !important;
    border-radius: 6px !important;
    padding: 2px !important;
}

/* Custom Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: #0a0a0a !important;
    border-right: 1px solid #1a1a1a !important;
}
[data-testid="stSidebar"] .stSlider label {
    color: #9ca3af !important;
    font-size: 0.82rem !important;
    font-family: 'Inter', sans-serif !important;
}

/* Bersihkan elemen dekoratif chrome bawaan lainnya */
#MainMenu, footer { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ── Mengurangi Padding Atas & Samping (Set ke 5%) ── */
.block-container {
    padding-top: 10px !important;      /* Memangkas jarak kosong di atas navbar */
    padding-bottom: 40px !important;
    padding-left: 5% !important;       /* Batas horizontal kiri 5% */
    padding-right: 5% !important;      /* Batas horizontal kanan 5% */
    max-width: 100% !important;
}

/* ── Navbar ── */
.navbar-brand {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
    color: #ffffff;
    letter-spacing: 0.02em;
}
.navbar-brand span { color: #6366f1; }

/* ── Streamlit widget overrides ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #6366f1 !important;
}
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #111 !important;
    border: 1px solid #222 !important;
    color: #e5e7eb !important;
    border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Buttons ── */
.stButton > button {
    background: #111 !important;
    color: #e5e7eb !important;
    border: 1px solid #262626 !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 9px 22px !important;
    transition: all 0.15s !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    border-color: #6366f1 !important;
    color: #a5b4fc !important;
    background: #0f0f1a !important;
}

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: #0d0d0d !important;
    border: 1px dashed #262626 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #6366f1 !important;
}

/* ── Alert boxes ── */
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.83rem !important; }
.stSuccess { background: #0a1a0f !important; border-left-color: #22c55e !important; }
.stError   { background: #1a0a0a !important; border-left-color: #ef4444 !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: #0d0d0d !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 8px !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #e5e7eb !important;
    font-size: 1.4rem !important;
}

/* ── Verdict ── */
.verdict-wrap {
    border-radius: 10px;
    padding: 24px 32px;
    text-align: center;
    margin: 16px 0;
}
.verdict-mirip  { background: #040e07; border: 1px solid #14532d; }
.verdict-tidak  { background: #100404; border: 1px solid #450a0a; }
.verdict-label  { font-size: 0.7rem; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 4px; }
.verdict-label.m { color: #4ade80; }
.verdict-label.t { color: #f87171; }
.verdict-score  { font-family: 'JetBrains Mono', monospace; font-size: 2.8rem; font-weight: 700; margin-bottom: 2px; }
.verdict-score.m { color: #4ade80; }
.verdict-score.t { color: #f87171; }
.verdict-meta   { font-size: 0.75rem; color: #374151; font-family: 'JetBrains Mono', monospace; }

/* ── Divider ── */
hr { border: none; border-top: 1px solid #141414; margin: 24px 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "latih"
if "n_components" not in st.session_state:
    st.session_state.n_components = 50
if "threshold_cos" not in st.session_state:
    st.session_state.threshold_cos = 0.60
if "pc_skip" not in st.session_state:
    st.session_state.pc_skip = 3

IMG_SIZE = (100, 100)

# ──────────────────────────────────────────────
#  SIDEBAR (Menu Parameter)
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;
                color:#6366f1;letter-spacing:0.1em;text-transform:uppercase;
                margin-top:10px;margin-bottom:20px;padding-bottom:10px;border-bottom:1px solid #1a1a1a;">
        ⚙️ Parameter PCA
    </div>
    """, unsafe_allow_html=True)

    st.session_state.n_components = st.slider(
        "Komponen PCA (k)", 10, 150, st.session_state.n_components, 5, key="sl_k"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.session_state.threshold_cos = st.slider(
        "Threshold Cosine", 0.0, 1.0, st.session_state.threshold_cos, 0.01, key="sl_thr"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    st.session_state.pc_skip = st.slider(
        "PC Skip (Pencahayaan)", 0, 10, st.session_state.pc_skip, 1, key="sl_skip"
    )

n_components  = st.session_state.n_components
threshold_cos = st.session_state.threshold_cos
pc_skip       = st.session_state.pc_skip

# ──────────────────────────────────────────────
#  NAVBAR MENGGUNAKAN COLUMNS
# ──────────────────────────────────────────────
col_nav = st.columns([3, 9]) 
with col_nav[0]:
    st.markdown("""
    <div style="padding: 14px 0 0 0;">
        <span class="navbar-brand">
            Face<span>Vec</span>
        </span>
    </div>
    """, unsafe_allow_html=True)

with col_nav[1]:
    nav1, nav2, nav3 = st.columns(3)
    with nav1:
        if st.button("Latih Model", key="nav_latih", use_container_width=True):
            st.session_state.active_tab = "latih"
            st.rerun()
    with nav2:
        if st.button("Bandingkan Wajah", key="nav_bandingkan", use_container_width=True):
            st.session_state.active_tab = "bandingkan"
            st.rerun()
    with nav3:
        if st.button("Kenali dari Database", key="nav_kenali", use_container_width=True):
            st.session_state.active_tab = "kenali"
            st.rerun()

st.markdown('<hr style="margin:8px 0 0 0;">', unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  UTILITAS & PENGOLAHAN MATEMATIS
# ──────────────────────────────────────────────
def save_uploaded_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        return f.name

def detect_face_preprocess(path: str):
    img_cv = cv2.imread(path)
    if img_cv is None:
        return None, None, None, False, "Tidak bisa membaca file."
    pil_img  = Image.open(path)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    cascade  = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(img_gray, 1.1, 5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None, pil_img, False, "Wajah tidak terdeteksi. Gunakan foto menghadap ke depan."
    x, y, w, h   = faces[0]
    face_gray     = img_gray[y:y+h, x:x+w]
    face_resized  = cv2.resize(face_gray, IMG_SIZE)
    clahe         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_resized  = clahe.apply(face_resized)
    face_norm     = face_resized.astype(np.float64) / 255.0
    vector        = face_norm.flatten()
    return vector, face_resized, pil_img, True, f"Terdeteksi ({x},{y}) {w}×{h}px"

def cosine_similarity(z1, z2) -> float:
    n1 = np.linalg.norm(z1); n2 = np.linalg.norm(z2)
    return 0.0 if (n1 == 0 or n2 == 0) else float(np.dot(z1, z2) / (n1 * n2))

def euclidean_distance(z1, z2) -> float:
    return float(np.linalg.norm(z1 - z2))

def ssim_simple(img1, img2) -> float:
    a, b   = img1.astype(np.float64), img2.astype(np.float64)
    mu1, mu2 = a.mean(), b.mean()
    s1,  s2  = a.std(),  b.std()
    cov      = np.mean((a - mu1) * (b - mu2))
    C1, C2   = (0.01*255)**2, (0.03*255)**2
    num = (2*mu1*mu2 + C1) * (2*cov + C2)
    den = (mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)
    return float((num / (den + 1e-10) + 1) / 2)

@st.cache_resource(show_spinner=False)
def load_lfw_pca(n_comp: int):
    lfw = fetch_lfw_people(min_faces_per_person=10, resize=0.4, color=False)
    X   = np.array([cv2.resize(img, IMG_SIZE).flatten() / 255.0 for img in lfw.images])
    mf  = np.mean(X, axis=0)
    pca = SklearnPCA(n_components=n_comp)
    pca.fit(X - mf)
    return pca, mf, X.shape[0]

def train_custom_pca(images_gray: list, n_comp: int):
    vecs = []
    for img in images_gray:
        fr  = cv2.resize(img, IMG_SIZE)
        cl  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        fr  = cl.apply(fr)
        vecs.append(fr.astype(np.float64).flatten() / 255.0)
    X   = np.array(vecs)
    mf  = np.mean(X, axis=0)
    k   = min(n_comp, X.shape[0] - 1)
    pca = SklearnPCA(n_components=k)
    pca.fit(X - mf)
    return pca, mf, X.shape[0]

# ── Visualisasi Matplotlib Eigenspace ──
DARK_FIG  = "#080808"
DARK_AX   = "#0d0d0d"
COL_ACC   = "#6366f1"
COL_B     = "#38bdf8"
COL_O     = "#f97316"

def _style_ax(ax, fig):
    fig.patch.set_facecolor(DARK_FIG)
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors="#374151", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#1a1a1a")
    ax.xaxis.label.set_color("#4b5563")
    ax.yaxis.label.set_color("#4b5563")
    ax.title.set_color("#9ca3af")

def fig_eigenfaces(pca_model, mean_face, n_show=8):
    Vk   = pca_model.components_.T
    n_ef = min(n_show, Vk.shape[1])
    cols = 4; rows = ((n_ef + 1) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.2, rows*2.4))
    axes = axes.flatten()
    fig.patch.set_facecolor(DARK_FIG)
    mf = mean_face.reshape(IMG_SIZE)
    mf = (mf - mf.min()) / (mf.max() - mf.min() + 1e-8)
    axes[0].imshow(mf, cmap="gray"); axes[0].set_title("mean", fontsize=7, color="#6b7280"); axes[0].axis("off")
    for i in range(n_ef):
        ef = Vk[:, i].reshape(IMG_SIZE)
        ef = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
        axes[i+1].imshow(ef, cmap="gray")
        axes[i+1].set_title(f"#{i+1}", fontsize=7, color="#6b7280")
        axes[i+1].axis("off")
    for j in range(n_ef+1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout(pad=0.5)
    return fig

def fig_cumvar(pca_model):
    ev     = pca_model.explained_variance_ratio_
    cumvar = np.cumsum(ev) * 100
    k      = len(ev)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    _style_ax(ax, fig)
    ax.plot(range(1, k+1), cumvar, color=COL_ACC, linewidth=1.5)
    ax.fill_between(range(1, k+1), cumvar, alpha=0.08, color=COL_ACC)
    ax.set_xlabel("Komponen (k)", fontsize=8)
    ax.set_ylabel("Variance kumulatif (%)", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.15, color="#222")
    plt.tight_layout()
    return fig

def fig_sv(pca_model):
    S   = np.sqrt(pca_model.explained_variance_)
    n   = min(20, len(S))
    fig, ax = plt.subplots(figsize=(8, 2.2))
    _style_ax(ax, fig)
    ax.bar(range(1, n+1), S[:n], color=COL_ACC, alpha=0.7, width=0.6)
    ax.set_ylabel("Singular Value", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.15, color="#222", axis="y")
    plt.tight_layout()
    return fig

def fig_proj(z1, z2, l1, l2):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    _style_ax(ax, fig)
    ax.scatter(z1[0], z1[1], c=COL_B, s=120, zorder=5, label=l1[:20])
    ax.scatter(z2[0], z2[1], c=COL_O, s=120, zorder=5, label=l2[:20])
    ax.plot([z1[0], z2[0]], [z1[1], z2[1]], "--", color="#374151", lw=1, alpha=0.6)
    ax.set_xlabel("PC 1", fontsize=8); ax.set_ylabel("PC 2", fontsize=8)
    ax.legend(fontsize=7, facecolor="#0d0d0d", labelcolor="white", edgecolor="#1a1a1a")
    ax.grid(True, alpha=0.1, color="#222")
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════
#  HERO HEADER SECTION
# ══════════════════════════════════════════════
active = st.session_state.active_tab

# padding diturunkan drastis agar teks langsung mepet ke atas dekat navbar
st.markdown("""
<div style="padding: 12px 0 12px;">
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;
                color:#6366f1;letter-spacing:0.12em;text-transform:uppercase;
                margin-bottom:8px;">
        PCA / SVD — Eigenfaces
    </div>
    <div style="font-size:clamp(1.8rem,3.5vw,2.3rem);font-weight:800;color:#fff;
                line-height:1.1;letter-spacing:-0.03em;margin-bottom:8px;">
        Deteksi Kemiripan Wajah secara Matematis
    </div>
    <div style="font-size:0.85rem;color:#4b5563;line-height:1.6;max-width:650px;">
        Proyeksi matriks wajah ke ruang dimensi rendah (<strong style="color:#6b7280;">eigenspace</strong>) 
        menggunakan SVD. Mengubah kompleksitas piksel gambar menjadi perbandingan vektor terarah.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr style="border-color:#141414;margin-bottom:20px;margin-top:10px;">', unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  TAB: LATIH MODEL
# ════════════════════════════════════════════════
if active == "latih":
    st.markdown('<div class="sec-eyebrow">TRAIN</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Latih Model PCA</div>', unsafe_allow_html=True)

    col_form, col_alur = st.columns([3, 2], gap="large")

    with col_alur:
        st.markdown("""
        <div style="background:#0d0d0d;border:1px solid #1a1a1a;border-radius:10px;padding:20px 22px;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#374151;
                        letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px;">Pipeline Alur Matematika</div>
            <div style="font-size:0.8rem;color:#6b7280;line-height:1.9;">
                <span style="color:#374151;font-family:'JetBrains Mono',monospace;">01</span>&nbsp;&nbsp;Upload matriks citra<br>
                <span style="color:#374151;font-family:'JetBrains Mono',monospace;">02</span>&nbsp;&nbsp;<code>Normalisasi & CLAHE</code><br>
                <span style="color:#374151;font-family:'JetBrains Mono',monospace;">03</span>&nbsp;&nbsp;<code>Deviasi Mean (X − μ)</code><br>
                <span style="color:#374151;font-family:'JetBrains Mono',monospace;">04</span>&nbsp;&nbsp;<code>SVD / Dekomposisi Eigen</code><br>
                <span style="color:#374151;font-family:'JetBrains Mono',monospace;">05</span>&nbsp;&nbsp;Eigenspace Terbentuk
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_form:
        use_lfw = st.toggle("Gunakan dataset LFW (unduh otomatis ~200MB)", value=False)

        if use_lfw:
            st.markdown('<div style="font-size:0.8rem;color:#6b7280;margin-bottom:12px;">Menggunakan kumpulan dataset wajah publik LFW People Dataset.</div>', unsafe_allow_html=True)
            if st.button("Latih dari LFW", key="train_lfw"):
                with st.spinner("Mengunduh & melatih PCA dari LFW..."):
                    try:
                        pca_m, mf, n_tr = load_lfw_pca(n_components)
                        st.session_state["pca_model"]   = pca_m
                        st.session_state["mean_face"]   = mf
                        st.session_state["n_train"]     = n_tr
                        st.session_state["model_source"] = f"LFW ({n_tr} foto)"
                        st.success(f"Model berhasil dilatih — k={n_components}")
                    except Exception as e:
                        st.error(str(e))
        else:
            uploaded_db = st.file_uploader(
                "Upload foto database wajah kustom (minimal 3 foto)",
                type=["jpg","jpeg","png"],
                accept_multiple_files=True,
                key="db_uploader"
            )

            if uploaded_db:
                st.markdown(f'<div style="font-size:0.78rem;color:#6b7280;margin:8px 0 12px;">{len(uploaded_db)} berkas siap diproses</div>', unsafe_allow_html=True)
                prev_cols = st.columns(min(len(uploaded_db), 5))
                db_grays  = []
                for i, uf in enumerate(uploaded_db):
                    pil_i  = Image.open(uf).convert("RGB")
                    cv_i   = cv2.cvtColor(np.array(pil_i), cv2.COLOR_RGB2BGR)
                    gray_i = cv2.cvtColor(cv_i, cv2.COLOR_BGR2GRAY)
                    db_grays.append(gray_i)
                    if i < 5:
                        prev_cols[i].image(pil_i, use_container_width=True)

                if st.button("Latih Model Kustom", key="train_custom"):
                    if len(db_grays) < 2:
                        st.error("Dibutuhkan minimal 2 sampel foto untuk ekstraksi matriks kovarian.")
                    else:
                        with st.spinner("Melatih model kustom..."):
                            try:
                                pca_m, mf, n_tr = train_custom_pca(db_grays, n_components)
                                st.session_state["pca_model"]   = pca_m
                                st.session_state["mean_face"]   = mf
                                st.session_state["n_train"]     = n_tr
                                st.session_state["model_source"] = f"{n_tr} foto custom"
                                st.success(f"Model kustom sukses diekstrak — k={pca_m.n_components_}")
                            except Exception as e:
                                st.error(str(e))

    if "pca_model" in st.session_state:
        pca_m = st.session_state["pca_model"]
        mf    = st.session_state["mean_face"]

        st.markdown('<hr>', unsafe_allow_html=True)
        st.markdown('<div class="sec-eyebrow">STATISTIK ELEMEN MATRIKS</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        m1.metric("Basis Data", st.session_state.get("model_source", "—"))
        m2.metric("Dimensi Komponen (k)", pca_m.n_components_)
        total_v = np.sum(pca_m.explained_variance_ratio_) * 100
        m3.metric("Explained Variance Retained", f"{total_v:.1f}%")

        c_ef, c_sv = st.columns(2, gap="large")
        with c_ef:
            st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin-bottom:8px;">Eigenfaces (Visualisasi Vektor Kolom V)</div>', unsafe_allow_html=True)
            st.pyplot(fig_eigenfaces(pca_m, mf))
        with c_sv:
            st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin-bottom:8px;">Singular Values (Skalar Magnitudo Σ)</div>', unsafe_allow_html=True)
            st.pyplot(fig_sv(pca_m))

        st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin:16px 0 8px;">Grafik Kumulatif Varian</div>', unsafe_allow_html=True)
        st.pyplot(fig_cumvar(pca_m))


# ════════════════════════════════════════════════
#  TAB: BANDINGKAN DUA WAJAH
# ════════════════════════════════════════════════
elif active == "bandingkan":
    st.markdown('<div class="sec-eyebrow">COMPARE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Bandingkan Dua Wajah</div>', unsafe_allow_html=True)

    if "pca_model" not in st.session_state:
        st.warning("Silakan latih atau muat model PCA terlebih dahulu di halaman utama.")
        st.stop()

    pca_m = st.session_state["pca_model"]
    mf    = st.session_state["mean_face"]

    up_col1, up_col2 = st.columns(2, gap="large")
    with up_col1:
        up1 = st.file_uploader("Upload Foto Pertama", type=["jpg","jpeg","png"], key="up1")
    with up_col2:
        up2 = st.file_uploader("Upload Foto Kedua", type=["jpg","jpeg","png"], key="up2")

    if up1 and up2:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_col = st.columns([2, 1, 2])
        with btn_col[1]:
            run = st.button("Uji Kemiripan", key="run_compare", use_container_width=True)

        if run:
            with st.spinner("Menghitung kedekatan matriks koordinat..."):
                p1 = save_uploaded_to_temp(up1)
                p2 = save_uploaded_to_temp(up2)
                vec1, face1, pil1, ok1, msg1 = detect_face_preprocess(p1)
                vec2, face2, pil2, ok2, msg2 = detect_face_preprocess(p2)

            st.markdown('<hr>', unsafe_allow_html=True)

            pc1, pc2 = st.columns(2, gap="large")
            with pc1:
                st.image(pil1, caption=up1.name, use_container_width=True)
                if ok1: st.success(msg1)
                else:   st.error(msg1)
            with pc2:
                st.image(pil2, caption=up2.name, use_container_width=True)
                if ok2: st.success(msg2)
                else:   st.error(msg2)

            if not ok1 or not ok2:
                st.error("Operasi dibatalkan karena kegagalan pemrosesan deteksi citra.")
                try: os.unlink(p1); os.unlink(p2)
                except: pass
                st.stop()

            z1f = pca_m.transform((vec1 - mf).reshape(1, -1))[0]
            z2f = pca_m.transform((vec2 - mf).reshape(1, -1))[0]
            k   = len(z1f)
            sk  = min(pc_skip, k - 2)
            z1  = z1f[sk:]; z2 = z2f[sk:]

            sim  = cosine_similarity(z1, z2)
            dist = euclidean_distance(z1, z2)
            ssim = ssim_simple(face1, face2)

            st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin:16px 0 8px;">Hasil Pemotongan ROI & Normalisasi Pencahayaan</div>', unsafe_allow_html=True)
            cr1, cr2 = st.columns(2, gap="large")
            cr1.image(face1, use_container_width=True, clamp=True)
            cr2.image(face2, use_container_width=True, clamp=True)

            st.markdown('<hr>', unsafe_allow_html=True)
            l1 = os.path.splitext(up1.name)[0]
            l2 = os.path.splitext(up2.name)[0]

            if sim >= threshold_cos:
                st.markdown(f"""
                <div class="verdict-wrap verdict-mirip">
                    <div class="verdict-label m">HASIL UJI: IDENTIK / MIRIP</div>
                    <div class="verdict-score m">{sim:.4f}</div>
                    <div class="verdict-meta">Cosine Similarity Melampaui Batas Batasan ({threshold_cos})</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-wrap verdict-tidak">
                    <div class="verdict-label t">HASIL UJI: BERBEDA / TIDAK MIRIP</div>
                    <div class="verdict-score t">{sim:.4f}</div>
                    <div class="verdict-meta">Cosine Similarity di Bawah Batas Batasan ({threshold_cos})</div>
                </div>""", unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Cosine Similarity", f"{sim:.4f}")
            m2.metric("Euclidean Distance", f"{dist:.2f}")
            m3.metric("Struktural SSIM Index", f"{ssim:.4f}")

            st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin:20px 0 8px;">Representasi Eigenspace 2 Dimensi Utama</div>', unsafe_allow_html=True)
            st.pyplot(fig_proj(z1, z2, l1, l2))

            try: os.unlink(p1); os.unlink(p2)
            except: pass


# ════════════════════════════════════════════════
#  TAB: KENALI DARI DATABASE
# ════════════════════════════════════════════════
elif active == "kenali":
    st.markdown('<div class="sec-eyebrow">RECOGNIZE</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Kenali Wajah dari Database</div>', unsafe_allow_html=True)

    if "pca_model" not in st.session_state:
        st.warning("Latih model terlebih dahulu di menu Latih Model.")
        st.stop()

    pca_m = st.session_state["pca_model"]
    mf    = st.session_state["mean_face"]

    kc1, kc2 = st.columns(2, gap="large")
    with kc1:
        db_files = st.file_uploader("Kumpulan Gambar Referensi Database", type=["jpg","jpeg","png"], accept_multiple_files=True)
    with kc2:
        qf = st.file_uploader("Gambar Kueri (Yang Ingin Dicari)", type=["jpg","jpeg","png"])

    if db_files and qf:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_c = st.columns([2, 1, 2])
        with btn_c[1]:
            run_r = st.button("Jalankan Rekognisi", key="run_recog", use_container_width=True)

        if run_r:
            with st.spinner("Mencari kecocokan spasial terdekat..."):
                db_entries = []
                for uf in db_files:
                    pt = save_uploaded_to_temp(uf)
                    vec, face, pil, ok, msg = detect_face_preprocess(pt)
                    label = os.path.splitext(uf.name)[0]
                    if ok:
                        zf  = pca_m.transform((vec - mf).reshape(1, -1))[0]
                        sk  = min(pc_skip, len(zf) - 2)
                        z   = zf[sk:]
                        db_entries.append({"label": label, "z": z, "face": face})
                    try: os.unlink(pt)
                    except: pass

                pq = save_uploaded_to_temp(qf)
                vq, fq, pq_img, okq, msgq = detect_face_preprocess(pq)
                try: os.unlink(pq)
                except: pass

            if not okq:
                st.error(f"Kegagalan kueri wajah: {msgq}")
                st.stop()
            if not db_entries:
                st.error("Database kosong atau wajah gagal terekstraksi.")
                st.stop()

            zqf = pca_m.transform((vq - mf).reshape(1, -1))[0]
            sk  = min(pc_skip, len(zqf) - 2)
            zq  = zqf[sk:]

            results = sorted(
                [(cosine_similarity(zq, e["z"]), e) for e in db_entries],
                key=lambda x: x[0], reverse=True
            )

            st.markdown('<hr>', unsafe_allow_html=True)

            rq1, rq2 = st.columns([1, 3], gap="large")
            rq1.image(pq_img, caption="Citra Kueri", use_container_width=True)
            best_sim, best_e = results[0]
            
            if best_sim >= threshold_cos:
                rq2.success(f"Sistem Berhasil Mengidentifikasi Objek: **{best_e['label']}** — Skor Kedekatan: {best_sim:.4f}")
            else:
                rq2.error(f"Akurasi rendah / Tidak terdaftar. Hasil terdekat: {best_sim:.4f} Berkemungkinan ({best_e['label']})")

            st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin:24px 0 12px;">Peringkat Matriks Kedekatan Kueri (Top 5)</div>', unsafe_allow_html=True)
            for rank, (sv, entry) in enumerate(results[:5], 1):
                r1, r2, r3, r4 = st.columns([1, 2, 4, 1], gap="small")
                r1.image(entry["face"], use_container_width=True, clamp=True)
                r2.markdown(f'<div style="font-size:0.72rem;color:#374151;font-family:JetBrains Mono,monospace;padding-top:4px;">Rangking {rank}</div>'
                            f'<div style="font-size:0.85rem;color:#e5e7eb;font-weight:600;">{entry["label"]}</div>', unsafe_allow_html=True)
                r3.progress(float(max(0, sv)))
                r4.metric("", f"{sv:.4f}")
