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
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
#  CSS — Sleek Black, no sidebar native
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
.main, .block-container {
    background-color: #080808 !important;
    color: #d4d4d4 !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Collapse native sidebar completely ── */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebarContent"] { display: none !important; }

/* ── Remove Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
.stDeployButton { display: none !important; }

/* ── Main content padding (TIDAK SEBECEK SEBELUMNYA GAK NEMPEL EDGE) ── */
.block-container {
    padding: 16px 48px 60px 48px !important;
    max-width: 100% !important;
}

/* ── Navbar ── */
.navbar {
    position: sticky;
    top: 0;
    z-index: 1000;
    background: rgba(8,8,8,0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid #1e1e1e;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 48px;
    height: 60px;
}
.navbar-brand {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.05rem;
    font-weight: 500;
    color: #ffffff;
    letter-spacing: 0.02em;
}
.navbar-brand span { color: #6366f1; }
.navbar-links {
    display: flex;
    align-items: center;
    gap: 4px;
}
.nav-btn {
    background: transparent;
    border: none;
    color: #6b7280;
    font-family: 'Inter', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    padding: 6px 16px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
    text-transform: uppercase;
}
.nav-btn:hover { color: #e5e7eb; background: #141414; }
.nav-btn.active { color: #ffffff; background: #141414; }

/* ── Custom Trigger Button Style ── */
div[data-testid="stBlock"] button[key="sidebar_trigger"] {
    background: transparent !important;
    border: none !important;
    color: #6366f1 !important;
    font-size: 1.3rem !important;
    padding: 0 !important;
    margin-top: -4px !important;
}

/* ── Page wrapper ── */
.page-wrap {
    max-width: 1100px;
    margin: 0 auto;
    padding: 48px 32px 80px;
}

/* ── Hero ── */
.hero {
    padding: 64px 0 48px;
    border-bottom: 1px solid #141414;
    margin-bottom: 48px;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    color: #6366f1;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 20px;
}
.hero-title {
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    letter-spacing: -0.03em;
    margin-bottom: 20px;
}
.hero-title em {
    font-style: normal;
    color: transparent;
    -webkit-text-stroke: 1px #6366f1;
}
.hero-sub {
    font-size: 0.95rem;
    color: #6b7280;
    line-height: 1.7;
    max-width: 560px;
}
.hero-sub strong { color: #9ca3af; font-weight: 500; }

/* ── Cards ── */
.card {
    background: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
.card-sm {
    background: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-radius: 8px;
    padding: 16px 20px;
}

/* ── Streamlit widget overrides ── */
[data-testid="stSlider"] {
    padding: 0 !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #6366f1 !important;
}
[data-testid="stSlider"] label {
    display: none !important;
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
    transition: border-color 0.2s;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: #6366f1 !important;
}

/* ── Verdict ── */
.verdict-wrap {
    border-radius: 10px;
    padding: 28px 32px;
    text-align: center;
    margin: 20px 0;
}
.verdict-mirip  { background: #040e07; border: 1px solid #14532d; }
.verdict-tidak  { background: #100404; border: 1px solid #450a0a; }
.verdict-label  { font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;
                  letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; }
.verdict-label.m { color: #4ade80; }
.verdict-label.t { color: #f87171; }
.verdict-score  { font-family: 'JetBrains Mono', monospace; font-size: 3rem;
                  font-weight: 700; letter-spacing: -0.02em; margin-bottom: 4px; }
.verdict-score.m { color: #4ade80; }
.verdict-score.t { color: #f87171; }
.verdict-meta   { font-size: 0.75rem; color: #374151; font-family: 'JetBrains Mono', monospace; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div { background: #6366f1 !important; }
hr { border: none; border-top: 1px solid #141414; margin: 32px 0; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  SESSION STATE
# ──────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "latih"
if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = False
if "n_components" not in st.session_state:
    st.session_state.n_components = 50
if "threshold_cos" not in st.session_state:
    st.session_state.threshold_cos = 0.60
if "pc_skip" not in st.session_state:
    st.session_state.pc_skip = 3

IMG_SIZE = (100, 100)

# ──────────────────────────────────────────────
#  NAVBAR (DIPERBARUI DENGAN TRIGGER LOGO)
# ──────────────────────────────────────────────
col_nav = st.columns([2.5, 5.5, 2])
with col_nav[0]:
    c_trigger, c_logo = st.columns([1, 4])
    with c_trigger:
        if st.button("◈", key="sidebar_trigger", help="Klik untuk membuka/tutup Parameter PCA"):
            st.session_state.sidebar_open = not st.session_state.sidebar_open
            st.rerun()
    with c_logo:
        st.markdown("""
        <div style="padding: 10px 0 0 0; select: none;">
            <span style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:500;color:#fff;">
                Face<span style="color:#6366f1;">Vec</span>
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

with col_nav[2]:
    st.markdown(f"""
    <div style="text-align: right; padding-top: 12px; font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #4b5563;">
        k={st.session_state.n_components} | thr={st.session_state.threshold_cos:.2f}
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr style="margin:8px 0 0 0; border-color:#1e1e1e;">', unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  UTILITAS & PLOT HELPERS
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
    cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
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

DARK_FIG  = "#080808"; DARK_AX   = "#0d0d0d"; COL_ACC   = "#6366f1"; COL_B     = "#38bdf8"; COL_O     = "#f97316"

def _style_ax(ax, fig):
    fig.patch.set_facecolor(DARK_FIG)
    ax.set_facecolor(DARK_AX)
    ax.tick_params(colors="#374151", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#1a1a1a")
    ax.xaxis.label.set_color("#4b5563"); ax.yaxis.label.set_color("#4b5563"); ax.title.set_color("#9ca3af")

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
    for j in range(n_ef+1, len(axes)): axes[j].axis("off")
    plt.tight_layout(pad=0.5)
    return fig

def fig_cumvar(pca_model):
    ev     = pca_model.explained_variance_ratio_
    cumvar = np.cumsum(ev) * 100
    k      = len(ev)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    _style_ax(ax, fig)
    ax.plot(range(1, k+1), cumvar, color=COL_ACC, linewidth=1.5)
    ax.fill_between(range(1, k+1), cumvar, alpha=0.08, color=COL_ACC)
    ax.set_xlabel("Komponen (k)", fontsize=8); ax.set_ylabel("Variance kumulatif (%)", fontsize=8)
    ax.set_title("Cumulative Explained Variance", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.15, color="#222")
    plt.tight_layout()
    return fig

def fig_sv(pca_model):
    S   = np.sqrt(pca_model.explained_variance_)
    n   = min(20, len(S))
    fig, ax = plt.subplots(figsize=(8, 2.5))
    _style_ax(ax, fig)
    ax.bar(range(1, n+1), S[:n], color=COL_ACC, alpha=0.7, width=0.6)
    ax.set_xlabel("Komponen ke-", fontsize=8); ax.set_ylabel("Singular Value", fontsize=8)
    ax.set_title(f"{n} Singular Values Terbesar", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.15, color="#222", axis="y")
    plt.tight_layout()
    return fig

def fig_proj(z1, z2, l1, l2):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    _style_ax(ax, fig)
    ax.scatter(z1[0], z1[1], c=COL_B, s=120, zorder=5, label=l1[:20])
    ax.scatter(z2[0], z2[1], c=COL_O, s=120, zorder=5, label=l2[:20])
    ax.plot([z1[0], z2[0]], [z1[1], z2[1]], "--", color="#374151", lw=1, alpha=0.6)
    ax.annotate(l1[:16], (z1[0], z1[1]), textcoords="offset points", xytext=(8, 5), fontsize=7, color=COL_B)
    ax.annotate(l2[:16], (z2[0], z2[1]), textcoords="offset points", xytext=(8, 5), fontsize=7, color=COL_O)
    ax.set_xlabel("PC 1", fontsize=8); ax.set_ylabel("PC 2", fontsize=8)
    ax.set_title("Proyeksi di Eigenspace", fontsize=9)
    ax.legend(fontsize=7, facecolor="#0d0d0d", labelcolor="white", edgecolor="#1a1a1a")
    ax.grid(True, alpha=0.1, color="#222")
    plt.tight_layout()
    return fig

# ──────────────────────────────────────────────
#  SIDEBAR LOGIC & LAYOUT SPLITTING
# ──────────────────────────────────────────────
if st.session_state.sidebar_open:
    col_sidebar, col_content = st.columns([1, 3.2], gap="large")
    with col_sidebar:
        st.markdown("""
        <div style="background:#0d0d0d; border:1px solid #1a1a1a; border-radius:10px; padding:20px; margin-top:24px;">
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#6366f1; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:16px; border-bottom:1px solid #141414; padding-bottom:8px;">
                ⚙️ Parameter PCA
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f'<div style="font-size:0.78rem;color:#9ca3af;margin-bottom:4px;">Komponen PCA (k) &nbsp;<span style="font-family:JetBrains Mono,monospace;color:#6366f1;">{st.session_state.n_components}</span></div>', unsafe_allow_html=True)
        new_k = st.slider("k", 10, 150, st.session_state.n_components, 5, label_visibility="collapsed", key="sd_k")
        if new_k != st.session_state.n_components:
            st.session_state.n_components = new_k
            st.rerun()

        st.markdown(f'<div style="font-size:0.78rem;color:#9ca3af;margin-bottom:4px;margin-top:16px;">Threshold Cosine &nbsp;<span style="font-family:JetBrains Mono,monospace;color:#6366f1;">{st.session_state.threshold_cos:.2f}</span></div>', unsafe_allow_html=True)
        new_thr = st.slider("thr", 0.0, 1.0, st.session_state.threshold_cos, 0.01, label_visibility="collapsed", key="sd_thr")
        if new_thr != st.session_state.threshold_cos:
            st.session_state.threshold_cos = new_thr
            st.rerun()

        st.markdown(f'<div style="font-size:0.78rem;color:#9ca3af;margin-bottom:4px;margin-top:16px;">PC Skip (pencahayaan) &nbsp;<span style="font-family:JetBrains Mono,monospace;color:#6366f1;">{st.session_state.pc_skip}</span></div>', unsafe_allow_html=True)
        new_skip = st.slider("skip", 0, 10, st.session_state.pc_skip, 1, label_visibility="collapsed", key="sd_skip")
        if new_skip != st.session_state.pc_skip:
            st.session_state.pc_skip = new_skip
            st.rerun()
else:
    col_content = st.container()

# Ambil nilai parameter aktif
n_components  = st.session_state.n_components
threshold_cos = st.session_state.threshold_cos
pc_skip       = st.session_state.pc_skip

# ──────────────────────────────────────────────
#  KONTEN UTAMA
# ──────────────────────────────────────────────
with col_content:
    active = st.session_state.active_tab

    # Hero Section
    st.markdown("""
    <div style="padding: 40px 0 32px;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:0.68rem; color:#6366f1;letter-spacing:0.12em;text-transform:uppercase; margin-bottom:16px;">
            PCA / SVD — Eigenfaces
        </div>
        <div style="font-size:clamp(1.8rem,4vw,2.6rem);font-weight:800;color:#fff; line-height:1.2;letter-spacing:-0.03em;margin-bottom:16px;">
            Deteksi Kemiripan<br>
            <span style="color:transparent;-webkit-text-stroke:1px #6366f1;">Wajah</span>
            &nbsp;secara Matematis
        </div>
        <div style="font-size:0.88rem;color:#6b7280;line-height:1.7;max-width:520px;">
            Setiap wajah diproyeksikan ke <strong style="color:#9ca3af;">eigenspace</strong> dan kemiripannya dihitung via <strong style="color:#9ca3af;">Cosine Similarity</strong>. 10.000 piksel diubah menjadi <strong style="color:#6366f1;">k dimensi</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr style="border-color:#141414;margin-bottom:32px;">', unsafe_allow_html=True)

    # ════════════════════════════════════════════════
    #  TAB: LATIH MODEL
    # ════════════════════════════════════════════════
    if active == "latih":
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#4b5563;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">TRAIN</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.3rem;font-weight:700;color:#e5e7eb;margin-bottom:24px;letter-spacing:-0.02em;">Latih Model PCA</div>', unsafe_allow_html=True)

        col_form, col_alur = st.columns([3, 2], gap="large")

        with col_alur:
            st.markdown("""
            <div style="background:#0d0d0d;border:1px solid #1a1a1a;border-radius:10px;padding:20px 22px;">
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#374151; letter-spacing:.1em;text-transform:uppercase;margin-bottom:14px;">Pipeline</div>
                <div style="font-size:0.8rem;color:#6b7280;line-height:1.9;">
                    <span style="color:#374151;font-family:'JetBrains Mono',monospace;">01</span>&nbsp;&nbsp; Upload foto referensi<br>
                    <span style="color:#374151;font-family:'JetBrains Mono',monospace;">02</span>&nbsp;&nbsp; <code style="font-family:'JetBrains Mono',monospace;color:#a5b4fc;background:#111827;padding:1px 5px;border-radius:3px;">grayscale → resize → CLAHE</code><br>
                    <span style="color:#374151;font-family:'JetBrains Mono',monospace;">03</span>&nbsp;&nbsp; <code style="font-family:'JetBrains Mono',monospace;color:#a5b4fc;background:#111827;padding:1px 5px;border-radius:3px;">Xc = X − mean_face</code><br>
                    <span style="color:#374151;font-family:'JetBrains Mono',monospace;">04</span>&nbsp;&nbsp; <code style="font-family:'JetBrains Mono',monospace;color:#a5b4fc;background:#111827;padding:1px 5px;border-radius:3px;">SVD(Xc) → eigenfaces</code><br>
                    <span style="color:#374151;font-family:'JetBrains Mono',monospace;">05</span>&nbsp;&nbsp; Eigenspace siap digunakan
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_form:
            use_lfw = st.toggle("Gunakan dataset LFW (unduh otomatis ~200MB)", value=False)

            if use_lfw:
                st.markdown('<div style="font-size:0.8rem;color:#6b7280;margin-bottom:12px;">Dataset LFW berisi ribuan foto wajah publik. Diunduh sekali, lalu ter-cache.</div>', unsafe_allow_html=True)
                if st.button("Latih dari LFW", key="train_lfw"):
                    with st.spinner("Mengunduh & melatih PCA dari LFW..."):
                        try:
                            pca_m, mf, n_tr = load_lfw_pca(n_components)
                            st.session_state["pca_model"]   = pca_m
                            st.session_state["mean_face"]   = mf
                            st.session_state["n_train"]     = n_tr
                            st.session_state["model_source"] = f"LFW ({n_tr} foto)"
                            st.success(f"Model dilatih dari {n_tr} foto LFW — k={n_components}")
                        except Exception as e:
                            st.error(str(e))
            else:
                uploaded_db = st.file_uploader(
                    "Upload foto database (minimal 3 foto, JPG/PNG)",
                    type=["jpg","jpeg","png"],
                    accept_multiple_files=True,
                    key="db_uploader"
                )

                if not uploaded_db:
                    st.markdown('<div style="font-size:0.8rem;color:#374151;margin-top:8px;">Upload foto untuk mulai pelatihan.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div style="font-size:0.78rem;color:#6b7280;margin:8px 0 12px;">{len(uploaded_db)} foto ter-upload</div>', unsafe_allow_html=True)
                    prev_cols = st.columns(min(len(uploaded_db), 5))
                    db_grays  = []
                    for i, uf in enumerate(uploaded_db):
                        pil_i  = Image.open(uf).convert("RGB")
                        cv_i   = cv2.cvtColor(np.array(pil_i), cv2.COLOR_RGB2BGR)
                        gray_i = cv2.cvtColor(cv_i, cv2.COLOR_BGR2GRAY)
                        db_grays.append(gray_i)
                        if i < 5:
                            prev_cols[i].image(pil_i, use_container_width=True)

                    if st.button("Latih Model", key="train_custom"):
                        if len(db_grays) < 2:
                            st.error("Minimal 2 foto untuk melatih PCA.")
                        else:
                            with st.spinner("Melatih model PCA..."):
                                try:
                                    pca_m, mf, n_tr = train_custom_pca(db_grays, n_components)
                                    st.session_state["pca_model"]   = pca_m
                                    st.session_state["mean_face"]   = mf
                                    st.session_state["n_train"]     = n_tr
                                    st.session_state["model_source"] = f"{n_tr} foto custom"
                                    st.success(f"Model dilatih dari {n_tr} foto — k={pca_m.n_components_}")
                                except Exception as e:
                                    st.error(str(e))

        if "pca_model" in st.session_state:
            pca_m = st.session_state["pca_model"]
            mf    = st.session_state["mean_face"]

            st.markdown('<hr style="border-color:#141414;margin:32px 0;">', unsafe_allow_html=True)
            st.markdown('<div style="font-size:0.78rem;color:#4b5563;font-family:JetBrains Mono,monospace;letter-spacing:.1em;text-transform:uppercase;margin-bottom:16px;">STATISTIK MODEL</div>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Sumber", st.session_state.get("model_source", "—"))
            m2.metric("Komponen k", pca_m.n_components_)
            total_v = np.sum(pca_m.explained_variance_ratio_) * 100
            m3.metric("Explained Variance", f"{total_v:.1f}%")

            c_ef, c_sv = st.columns(2, gap="large")
            with c_ef:
                st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin-bottom:8px;">Eigenfaces</div>', unsafe_allow_html=True)
                st.pyplot(fig_eigenfaces(pca_m, mf))
            with c_sv:
                st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin-bottom:8px;">Singular Values</div>', unsafe_allow_html=True)
                st.pyplot(fig_sv(pca_m))

            st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin:16px 0 8px;">Cumulative Variance</div>', unsafe_allow_html=True)
            st.pyplot(fig_cumvar(pca_m))

    # ════════════════════════════════════════════════
    #  TAB: BANDINGKAN DUA WAJAH (FIXED COGNITION LOGIC)
    # ════════════════════════════════════════════════
    elif active == "bandingkan":
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#4b5563;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">COMPARE</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.3rem;font-weight:700;color:#e5e7eb;margin-bottom:24px;letter-spacing:-0.02em;">Bandingkan Dua Wajah</div>', unsafe_allow_html=True)

        if "pca_model" not in st.session_state:
            st.warning("Latih model terlebih dahulu di menu Latih Model.")
            st.stop()

        pca_m = st.session_state["pca_model"]
        mf    = st.session_state["mean_face"]

        st.markdown('<div style="font-size:0.83rem;color:#4b5563;margin-bottom:20px;">Upload dua foto wajah — misalnya foto masa kecil vs dewasa.</div>', unsafe_allow_html=True)

        up_col1, up_col2 = st.columns(2, gap="large")
        with up_col1:
            st.markdown('<div style="font-size:0.75rem;color:#6b7280;margin-bottom:6px;">Foto 1</div>', unsafe_allow_html=True)
            up1 = st.file_uploader("Foto 1", type=["jpg","jpeg","png"], key="up1", label_visibility="collapsed")
        with up_col2:
            st.markdown('<div style="font-size:0.75rem;color:#6b7280;margin-bottom:6px;">Foto 2</div>', unsafe_allow_html=True)
            up2 = st.file_uploader("Foto 2", type=["jpg","jpeg","png"], key="up2", label_visibility="collapsed")

        if not up1 or not up2:
            st.markdown('<div style="font-size:0.8rem;color:#374151;margin-top:8px;">Upload kedua foto untuk memulai perbandingan.</div>', unsafe_allow_html=True)
        else:
            st.markdown("<br>", unsafe_allow_html=True)
            btn_col = st.columns([2, 1, 2])
            with btn_col[1]:
                run = st.button("Bandingkan", key="run_compare", use_container_width=True)

            if run:
                with st.spinner("Memproses..."):
                    p1 = save_uploaded_to_temp(up1)
                    p2 = save_uploaded_to_temp(up2)
                    vec1, face1, pil1, ok1, msg1 = detect_face_preprocess(p1)
                    vec2, face2, pil2, ok2, msg2 = detect_face_preprocess(p2)

                st.markdown('<hr style="border-color:#141414;margin:24px 0;">', unsafe_allow_html=True)

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
                    st.error("Perbandingan tidak bisa dilanjutkan.")
                    try: os.unlink(p1); os.unlink(p2)
                    except: pass
                    st.stop()

                # --- PENERAPAN WHITENING UNTUK MENCEGAH FALSE POSITIVES ---
                scaling = np.sqrt(pca_m.explained_variance_ + 1e-8)
                z1f = pca_m.transform((vec1 - mf).reshape(1, -1))[0] / scaling
                z2f = pca_m.transform((vec2 - mf).reshape(1, -1))[0] / scaling
                
                k   = len(z1f)
                sk  = min(pc_skip, k - 2)
                z1  = z1f[sk:]; z2 = z2f[sk:]

                sim  = cosine_similarity(z1, z2)
                dist = euclidean_distance(z1, z2)
                ssim = ssim_simple(face1, face2)

                st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin:20px 0 8px;">Wajah ter-crop (100×100, CLAHE)</div>', unsafe_allow_html=True)
                cr1, cr2 = st.columns(2, gap="large")
                cr1.image(face1, use_container_width=True, clamp=True)
                cr2.image(face2, use_container_width=True, clamp=True)

                st.markdown('<hr style="border-color:#141414;margin:24px 0;">', unsafe_allow_html=True)
                l1 = os.path.splitext(up1.name)[0]
                l2 = os.path.splitext(up2.name)[0]

                if sim >= threshold_cos:
                    st.markdown(f"""
                    <div class="verdict-wrap verdict-mirip">
                        <div class="verdict-label m">MIRIP</div>
                        <div class="verdict-score m">{sim:.4f}</div>
                        <div class="verdict-meta">cosine similarity &nbsp;|&nbsp; threshold {threshold_cos}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="verdict-wrap verdict-tidak">
                        <div class="verdict-label t">TIDAK MIRIP</div>
                        <div class="verdict-score t">{sim:.4f}</div>
                        <div class="verdict-meta">cosine similarity &nbsp;|&nbsp; threshold {threshold_cos}</div>
                    </div>""", unsafe_allow_html=True)

                m1, m2, m3 = st.columns(3)
                m1.metric("Cosine Similarity", f"{sim:.4f}", delta=f"{'di atas' if sim >= threshold_cos else 'di bawah'} threshold")
                m2.metric("Euclidean Distance", f"{dist:.2f}")
                m3.metric("SSIM", f"{ssim:.4f}")

                st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin:20px 0 8px;">Proyeksi di Eigenspace</div>', unsafe_allow_html=True)
                st.pyplot(fig_proj(z1, z2, l1, l2))

                with st.expander("Detail Teknis"):
                    st.markdown(f"""
    | Parameter | Nilai |
    |---|---|
    | Sumber model | {st.session_state.get('model_source','—')} |
    | Komponen PCA (k) | {pca_m.n_components_} |
    | Dimensi asli | {IMG_SIZE[0]*IMG_SIZE[1]} px |
    | Dimensi setelah PCA (Whitened) | {k - sk} |
    | PC di-skip | {sk} |
    | z₁ (5 pertama) | `{z1[:5].round(4).tolist()}` |
    | z₂ (5 pertama) | `{z2[:5].round(4).tolist()}` |
                    """)

                try: os.unlink(p1); os.unlink(p2)
                except: pass

    # ════════════════════════════════════════════════
    #  TAB: KENALI DARI DATABASE (FIXED COGNITION LOGIC)
    # ════════════════════════════════════════════════
    elif active == "kenali":
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.68rem;color:#4b5563;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">RECOGNIZE</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.3rem;font-weight:700;color:#e5e7eb;margin-bottom:24px;letter-spacing:-0.02em;">Kenali Wajah dari Database</div>', unsafe_allow_html=True)

        if "pca_model" not in st.session_state:
            st.warning("Latih model terlebih dahulu di menu Latih Model.")
            st.stop()

        pca_m = st.session_state["pca_model"]
        mf    = st.session_state["mean_face"]

        st.markdown('<div style="font-size:0.83rem;color:#4b5563;margin-bottom:20px;">Upload foto database (nama file = label orang), lalu upload satu foto query.</div>', unsafe_allow_html=True)

        kc1, kc2 = st.columns(2, gap="large")
        with kc1:
            st.markdown('<div style="font-size:0.75rem;color:#6b7280;margin-bottom:6px;">Foto Database</div>', unsafe_allow_html=True)
            db_files = st.file_uploader("Database", type=["jpg","jpeg","png"], accept_multiple_files=True, key="recog_db", label_visibility="collapsed")
        with kc2:
            st.markdown('<div style="font-size:0.75rem;color:#6b7280;margin-bottom:6px;">Foto Query</div>', unsafe_allow_html=True)
            qf = st.file_uploader("Query", type=["jpg","jpeg","png"], key="recog_query", label_visibility="collapsed")

        if db_files and qf:
            st.markdown("<br>", unsafe_allow_html=True)
            btn_c = st.columns([2, 1, 2])
            with btn_c[1]:
                run_r = st.button("Kenali", key="run_recog", use_container_width=True)

            if run_r:
                with st.spinner("Memproses..."):
                    scaling = np.sqrt(pca_m.explained_variance_ + 1e-8)
                    db_entries = []
                    for uf in db_files:
                        pt = save_uploaded_to_temp(uf)
                        vec, face, pil, ok, msg = detect_face_preprocess(pt)
                        label = os.path.splitext(uf.name)[0]
                        if ok:
                            zf  = pca_m.transform((vec - mf).reshape(1, -1))[0] / scaling
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
                    st.error(f"Wajah tidak terdeteksi di foto query: {msgq}")
                    st.stop()
                if not db_entries:
                    st.error("Tidak ada wajah terdeteksi di database.")
                    st.stop()

                zqf = pca_m.transform((vq - mf).reshape(1, -1))[0] / scaling
                sk  = min(pc_skip, len(zqf) - 2)
                zq  = zqf[sk:]

                results = sorted([(cosine_similarity(zq, e["z"]), e) for e in db_entries], key=lambda x: x[0], reverse=True)

                st.markdown('<hr style="border-color:#141414;margin:24px 0;">', unsafe_allow_html=True)

                rq1, rq2 = st.columns([1, 3], gap="large")
                rq1.image(pq_img, caption="Query", use_container_width=True)
                best_sim, best_e = results[0]
                if best_sim >= threshold_cos:
                    rq2.success(f"Dikenali sebagai **{best_e['label']}** — skor {best_sim:.4f}")
                else:
                    rq2.error(f"Tidak ada yang cukup mirip. Skor tertinggi: {best_sim:.4f} ({best_e['label']})")

                st.markdown('<div style="font-size:0.75rem;color:#4b5563;margin:24px 0 12px;">Ranking Kemiripan</div>', unsafe_allow_html=True)
                for rank, (sv, entry) in enumerate(results[:5], 1):
                    r1, r2, r3, r4 = st.columns([1, 2, 4, 1], gap="small")
                    r1.image(entry["face"], use_container_width=True, clamp=True)
                    r2.markdown(f'<div style="font-size:0.72rem;color:#374151;font-family:JetBrains Mono,monospace;padding-top:4px;">#{rank}</div>'
                                f'<div style="font-size:0.85rem;color:#e5e7eb;font-weight:600;">{entry["label"]}</div>', unsafe_allow_html=True)
                    r3.progress(float(max(0, sv)))
                    verdict_t = "MIRIP" if sv >= threshold_cos else "—"
                    r4.metric("", f"{sv:.4f}", delta=verdict_t)
        else:
            if not db_files:
                st.markdown('<div style="font-size:0.8rem;color:#374151;">Upload foto database terlebih dahulu.</div>', unsafe_allow_html=True)
            if not qf:
                st.markdown('<div style="font-size:0.8rem;color:#374151;">Upload foto query yang ingin dikenali.</div>', unsafe_allow_html=True)
