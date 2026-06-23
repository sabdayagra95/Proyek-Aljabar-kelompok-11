"""
=============================================================
  DETEKSI KEMIRIPAN WAJAH — PCA / SVD  (Streamlit App)
  Eigenspace dari Dataset LFW + Upload Foto Sendiri

  Jalankan:
      streamlit run app.py

  Requirements:
      pip install streamlit numpy opencv-python-headless
                  matplotlib pillow scikit-learn
=============================================================
"""

import os
import sys
import tempfile
import warnings

warnings.filterwarnings("ignore")

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA as SklearnPCA

# ──────────────────────────────────────────────
#  KONFIGURASI HALAMAN
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Kemiripan Wajah",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  CUSTOM CSS — Dark Theme mirip screenshot
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global dark background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f0f1a;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #1a1a2e;
    border-right: 1px solid #2a2a4a;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}

/* ── Header gradien ungu ── */
.header-box {
    background: linear-gradient(135deg, #1a1a3e 0%, #2d1b69 50%, #1a1a3e 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 24px;
    border: 1px solid #3d2b8a;
}
.header-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #e040fb;
    font-family: 'Courier New', monospace;
    margin: 0 0 12px 0;
}
.header-title span.icon { margin-right: 12px; }
.header-subtitle {
    color: #b0b0cc;
    font-size: 1rem;
    line-height: 1.6;
}
.header-subtitle strong { color: #e0e0ff; }

/* ── Tab styling ── */
[data-testid="stTabs"] button {
    background: transparent;
    color: #888 !important;
    border-bottom: 2px solid transparent;
    font-weight: 600;
    padding: 8px 20px;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #f97316 !important;
    border-bottom: 2px solid #f97316 !important;
}

/* ── Kartu info alur ── */
.alur-box {
    background-color: #1a1a2e;
    border: 1px solid #2a2a5a;
    border-radius: 10px;
    padding: 18px 20px;
    font-size: 0.82rem;
    color: #b0b8d0;
    line-height: 1.9;
}
.alur-box code {
    background: #2a2a4a;
    color: #e040fb;
    padding: 1px 5px;
    border-radius: 4px;
    font-size: 0.78rem;
}
.alur-title {
    color: #ffffff;
    font-weight: 700;
    margin-bottom: 10px;
    font-size: 0.9rem;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1e1e30;
    border-radius: 10px;
    padding: 14px 18px;
    border: 1px solid #2a2a5a;
}
[data-testid="stMetricValue"] { color: #e040fb !important; }
[data-testid="stMetricLabel"] { color: #888 !important; }

/* ── Info / warning / success boxes ── */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6d28d9, #9333ea);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 700;
    font-size: 0.95rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #7c3aed, #a855f7);
    transform: translateY(-1px);
}

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div {
    background: #e040fb !important;
}

/* ── Section label ── */
.section-label {
    font-family: 'Courier New', monospace;
    color: #f97316;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.section-heading {
    font-size: 1.5rem;
    font-weight: 800;
    color: #ffffff;
    margin-bottom: 16px;
}

/* ── Verdict cards ── */
.verdict-mirip {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #059669;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.verdict-tidak {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
}
.verdict-text {
    font-size: 1.8rem;
    font-weight: 900;
    margin-bottom: 6px;
}
.verdict-sim {
    font-size: 2.5rem;
    font-weight: 900;
    font-family: 'Courier New', monospace;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  KONFIGURASI DEFAULT
# ──────────────────────────────────────────────
IMG_SIZE = (100, 100)

# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameter PCA")
    st.markdown("---")

    n_components = st.slider(
        "Jumlah komponen PCA (k)",
        min_value=10, max_value=150, value=50, step=5,
        help="Jumlah eigenface yang digunakan. Lebih banyak = lebih akurat tapi lebih lambat."
    )
    threshold_cos = st.slider(
        "Threshold Cosine Similarity",
        min_value=0.0, max_value=1.0, value=0.60, step=0.01,
        help="Nilai ≥ threshold → MIRIP. Nilai < threshold → TIDAK MIRIP."
    )
    pc_skip = st.slider(
        "PC pertama yang di-skip",
        min_value=0, max_value=10, value=3, step=1,
        help="PC awal menangkap pencahayaan global, bukan identitas. Direkomendasikan skip 1–3."
    )

    st.markdown("---")

    # ── Ringkasan Alur ──
    st.markdown('<div class="alur-title">📋 Ringkasan Alur</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="alur-box">
    1. Upload ≥2 foto referensi (database)<br>
    2. Upload 2 foto yang akan dibandingkan<br>
    3. Preprocessing: <code>grayscale → resize → normalize → flatten</code><br>
    4. <code>Xc = X − mean_face</code> (centering)<br>
    5. <code>SVD(Xc) → eigenfaces</code><br>
    6. <code>Z = Xc · Vk</code> (proyeksi)<br>
    7. Hitung <code>cosine_sim</code> &amp; <code>euclidean_dist</code><br>
    8. Bandingkan dengan threshold
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Implementasi PCA/SVD · Eigenfaces · Deteksi Kemiripan Wajah")

# ──────────────────────────────────────────────
#  HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <div class="header-title">
        <span class="icon">🧠</span> Deteksi Kemiripan Wajah
    </div>
    <div class="header-subtitle">
        Implementasi <strong>PCA / SVD (Eigenfaces)</strong> untuk membandingkan dua wajah.<br>
        Setiap gambar direduksi dari <strong>10.000 piksel → k dimensi PCA</strong>,
        lalu kemiripan dihitung via <strong>Cosine Similarity</strong> dan <strong>Euclidean Distance</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  FUNGSI UTILITAS
# ──────────────────────────────────────────────
def save_uploaded_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded_file.read())
        return f.name


def detect_face_preprocess(path: str):
    """
    Deteksi wajah (Haar Cascade), crop, CLAHE, resize ke IMG_SIZE.
    Returns (vector, face_resized_uint8, pil_img, detected:bool, msg:str)
    """
    img_cv = cv2.imread(path)
    if img_cv is None:
        return None, None, None, False, "Tidak bisa membaca file gambar."

    pil_img = Image.open(path)
    img_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(
        img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        return None, None, pil_img, False, "Wajah tidak terdeteksi. Coba foto yang lebih jelas / menghadap ke depan."

    x, y, w, h = faces[0]
    face_gray = img_gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face_gray, IMG_SIZE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_resized = clahe.apply(face_resized)

    face_norm = face_resized.astype(np.float64) / 255.0
    vector = face_norm.flatten()

    return vector, face_resized, pil_img, True, f"Wajah terdeteksi — area ({x},{y}) {w}×{h}px"


def cosine_similarity(z1, z2) -> float:
    n1 = np.linalg.norm(z1)
    n2 = np.linalg.norm(z2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(z1, z2) / (n1 * n2))


def euclidean_distance(z1, z2) -> float:
    return float(np.linalg.norm(z1 - z2))


@st.cache_resource(show_spinner=False)
def load_lfw_pca(n_comp: int):
    """Download & latih PCA dari dataset LFW (di-cache Streamlit)."""
    lfw = fetch_lfw_people(min_faces_per_person=10, resize=0.4, color=False)
    X = np.array([
        cv2.resize(img, IMG_SIZE).flatten() / 255.0
        for img in lfw.images
    ])
    mean_face = np.mean(X, axis=0)
    Xc = X - mean_face
    pca = SklearnPCA(n_components=n_comp)
    pca.fit(Xc)
    return pca, mean_face, X.shape[0]


def train_custom_pca(images_gray: list, n_comp: int):
    """Latih PCA dari foto yang diupload user."""
    vecs = []
    for img in images_gray:
        face_resized = cv2.resize(img, IMG_SIZE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_resized = clahe.apply(face_resized)
        vecs.append(face_resized.astype(np.float64).flatten() / 255.0)

    X = np.array(vecs)
    mean_face = np.mean(X, axis=0)
    Xc = X - mean_face
    k = min(n_comp, X.shape[0] - 1)
    pca = SklearnPCA(n_components=k)
    pca.fit(Xc)
    return pca, mean_face, X.shape[0]


def fig_eigenfaces(pca_model, mean_face, n_show=8):
    Vk = pca_model.components_.T
    n_ef = min(n_show, Vk.shape[1])
    cols = 4
    rows = ((n_ef + 1) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8),
                             facecolor="#0f0f1a")
    axes = axes.flatten()

    mf_show = mean_face.reshape(IMG_SIZE)
    mf_norm = (mf_show - mf_show.min()) / (mf_show.max() - mf_show.min() + 1e-8)
    axes[0].imshow(mf_norm, cmap="gray")
    axes[0].set_title("Mean Face", fontsize=8, color="white")
    axes[0].axis("off")

    for i in range(n_ef):
        ef = Vk[:, i].reshape(IMG_SIZE)
        ef_norm = (ef - ef.min()) / (ef.max() - ef.min() + 1e-8)
        axes[i + 1].imshow(ef_norm, cmap="gray")
        axes[i + 1].set_title(f"Eigenface #{i+1}", fontsize=8, color="white")
        axes[i + 1].axis("off")

    for j in range(n_ef + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Mean Face & {n_ef} Eigenfaces",
                 fontsize=11, fontweight="bold", color="white")
    plt.tight_layout()
    return fig


def fig_cumulative_variance(pca_model):
    ev = pca_model.explained_variance_ratio_
    cumvar = np.cumsum(ev) * 100
    k = len(ev)

    fig, ax = plt.subplots(figsize=(8, 3.5), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.plot(range(1, k + 1), cumvar, color="#e040fb", linewidth=2)
    ax.fill_between(range(1, k + 1), cumvar, alpha=0.15, color="#e040fb")
    ax.set_xlabel("Jumlah Komponen (k)", color="#aaa", fontsize=9)
    ax.set_ylabel("Cumulative Variance (%)", color="#aaa", fontsize=9)
    ax.set_title("Cumulative Explained Variance", color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#888")
    ax.spines[:].set_color("#2a2a5a")
    ax.grid(True, linestyle="--", alpha=0.3, color="#444")
    plt.tight_layout()
    return fig


def fig_projection(z1, z2, label1, label2):
    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0f0f1a")
    ax.set_facecolor("#12122a")
    ax.scatter(z1[0], z1[1], c="#38bdf8", s=200, zorder=5, label=label1[:20])
    ax.scatter(z2[0], z2[1], c="#f97316", s=200, zorder=5, label=label2[:20])
    ax.plot([z1[0], z2[0]], [z1[1], z2[1]], "w--", linewidth=1.5, alpha=0.5)
    ax.annotate(label1[:18], (z1[0], z1[1]),
                textcoords="offset points", xytext=(8, 6),
                fontsize=8, color="#38bdf8")
    ax.annotate(label2[:18], (z2[0], z2[1]),
                textcoords="offset points", xytext=(8, 6),
                fontsize=8, color="#f97316")
    ax.set_xlabel("PC 1", color="#888", fontsize=8)
    ax.set_ylabel("PC 2", color="#888", fontsize=8)
    ax.set_title("Proyeksi di Eigenspace", color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#666")
    ax.spines[:].set_color("#2a2a5a")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white", edgecolor="#3a3a6a")
    ax.grid(True, alpha=0.2, color="#444")
    plt.tight_layout()
    return fig


def fig_singular_values(pca_model, n_show=20):
    S = np.sqrt(pca_model.explained_variance_ * 1)
    n = min(n_show, len(S))
    fig, ax = plt.subplots(figsize=(8, 3), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.bar(range(1, n + 1), S[:n], color="#7c3aed", alpha=0.85)
    ax.set_xlabel("Komponen ke-", color="#aaa", fontsize=9)
    ax.set_ylabel("Singular Value (σ)", color="#aaa", fontsize=9)
    ax.set_title(f"{n} Singular Values Terbesar", color="white", fontsize=10, fontweight="bold")
    ax.tick_params(colors="#888")
    ax.spines[:].set_color("#2a2a5a")
    ax.grid(True, linestyle="--", alpha=0.3, color="#444")
    plt.tight_layout()
    return fig


# ──────────────────────────────────────────────
#  TABS UTAMA
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🏋️ Latih Model (Database)",
    "🔍 Bandingkan Dua Wajah",
    "🌐 Kenali dari Database",
])

# ════════════════════════════════════════════════
#  TAB 1 — LATIH MODEL
# ════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-label">TRAIN DATA</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Upload Foto Referensi (Database Wajah) 🔗</div>',
                unsafe_allow_html=True)

    st.info(
        "Upload **minimal 3 foto** sebagai data latih PCA. "
        "Bisa foto satu orang atau banyak orang — sistem akan mempelajari *variasi* wajah "
        "dan membangun **eigenfaces**."
    )

    use_lfw = st.toggle("Gunakan dataset LFW (otomatis, ~200MB unduh pertama kali)", value=False)

    if use_lfw:
        st.warning("⚠️ Dataset LFW akan diunduh saat pertama kali (~200MB). Selanjutnya ter-cache.")
        if st.button("🚀 Latih PCA dari Dataset LFW"):
            with st.spinner("Mengunduh & melatih PCA dari LFW... (mungkin 1–2 menit pertama kali)"):
                try:
                    pca_model, mean_face, n_train = load_lfw_pca(n_components)
                    st.session_state["pca_model"] = pca_model
                    st.session_state["mean_face"] = mean_face
                    st.session_state["n_train"] = n_train
                    st.session_state["model_source"] = "LFW"
                    st.success(f"✅ Model PCA berhasil dilatih dari **{n_train} foto LFW** dengan **k={n_components}** komponen!")
                except Exception as e:
                    st.error(f"❌ Gagal melatih model: {e}")
    else:
        uploaded_db = st.file_uploader(
            "Upload foto database (JPG/PNG)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="db_uploader"
        )

        if not uploaded_db:
            st.warning("👆 Upload foto database terlebih dahulu.")
        else:
            st.success(f"✅ {len(uploaded_db)} foto ter-upload.")

            # Preview foto database
            cols_prev = st.columns(min(len(uploaded_db), 6))
            db_grays = []
            for i, uf in enumerate(uploaded_db):
                img_pil = Image.open(uf).convert("RGB")
                img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                db_grays.append(gray)
                if i < 6:
                    cols_prev[i].image(img_pil, caption=uf.name[:15], use_container_width=True)

            if st.button("🚀 Latih PCA dari Foto yang Diupload"):
                if len(db_grays) < 2:
                    st.error("❌ Butuh minimal 2 foto untuk melatih PCA.")
                else:
                    with st.spinner("Melatih model PCA..."):
                        try:
                            pca_model, mean_face, n_train = train_custom_pca(db_grays, n_components)
                            st.session_state["pca_model"] = pca_model
                            st.session_state["mean_face"] = mean_face
                            st.session_state["n_train"] = n_train
                            st.session_state["model_source"] = f"{n_train} foto custom"
                            st.success(f"✅ Model PCA berhasil dilatih dari **{n_train} foto** dengan **k={pca_model.n_components_}** komponen!")
                        except Exception as e:
                            st.error(f"❌ Gagal: {e}")

    # Tampilkan info model jika sudah dilatih
    if "pca_model" in st.session_state:
        pca_model = st.session_state["pca_model"]
        mean_face = st.session_state["mean_face"]
        n_train   = st.session_state["n_train"]
        src       = st.session_state.get("model_source", "?")

        st.markdown("---")
        st.markdown("### 📊 Statistik Model")

        c1, c2, c3 = st.columns(3)
        c1.metric("Sumber Data", src)
        c2.metric("Komponen PCA (k)", pca_model.n_components_)
        total_var = np.sum(pca_model.explained_variance_ratio_) * 100
        c3.metric("Explained Variance", f"{total_var:.1f}%")

        col_ef, col_sv = st.columns(2)
        with col_ef:
            st.pyplot(fig_eigenfaces(pca_model, mean_face, n_show=8))
        with col_sv:
            st.pyplot(fig_singular_values(pca_model))

        st.pyplot(fig_cumulative_variance(pca_model))

        st.success("✅ Model siap digunakan! Buka tab **Bandingkan Dua Wajah** untuk mulai.")
    else:
        st.info("ℹ️ Model belum dilatih. Latih dulu di atas, atau gunakan dataset LFW.")


# ════════════════════════════════════════════════
#  TAB 2 — BANDINGKAN DUA WAJAH
# ════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-label">COMPARE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Bandingkan Dua Wajah 🔍</div>',
                unsafe_allow_html=True)

    if "pca_model" not in st.session_state:
        st.warning("⚠️ Latih model terlebih dahulu di tab **Latih Model (Database)**.")
        st.stop()

    pca_model = st.session_state["pca_model"]
    mean_face = st.session_state["mean_face"]

    st.info("Upload dua foto wajah yang ingin dibandingkan (contoh: foto masa kecil vs dewasa).")

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        st.markdown("**📸 Foto 1 (misal: Masa Kecil)**")
        up1 = st.file_uploader("Upload Foto 1", type=["jpg","jpeg","png"], key="up1")
    with col_up2:
        st.markdown("**📸 Foto 2 (misal: Masa Dewasa)**")
        up2 = st.file_uploader("Upload Foto 2", type=["jpg","jpeg","png"], key="up2")

    if not up1 or not up2:
        st.warning("👆 Upload kedua foto untuk mulai perbandingan.")
    else:
        if st.button("🔍 Bandingkan Sekarang!", use_container_width=True):
            with st.spinner("Memproses gambar..."):
                path1 = save_uploaded_to_temp(up1)
                path2 = save_uploaded_to_temp(up2)

                vec1, face1, pil1, ok1, msg1 = detect_face_preprocess(path1)
                vec2, face2, pil2, ok2, msg2 = detect_face_preprocess(path2)

            # Preview input
            st.markdown("### 🖼️ Gambar Input")
            c1, c2 = st.columns(2)
            with c1:
                st.image(pil1, caption=f"Foto 1: {up1.name}", use_container_width=True)
                if ok1:
                    st.success(f"✅ {msg1}")
                else:
                    st.error(f"❌ {msg1}")
            with c2:
                st.image(pil2, caption=f"Foto 2: {up2.name}", use_container_width=True)
                if ok2:
                    st.success(f"✅ {msg2}")
                else:
                    st.error(f"❌ {msg2}")

            if not ok1 or not ok2:
                st.error("❌ Perbandingan tidak bisa dilanjutkan karena wajah tidak terdeteksi di salah satu foto.")
                st.stop()

            # ── Proyeksi ke eigenspace ──
            z1_full = pca_model.transform((vec1 - mean_face).reshape(1, -1))[0]
            z2_full = pca_model.transform((vec2 - mean_face).reshape(1, -1))[0]

            k = len(z1_full)
            skip = min(pc_skip, k - 2)
            z1 = z1_full[skip:]
            z2 = z2_full[skip:]

            # ── Hitung similarity ──
            sim  = cosine_similarity(z1, z2)
            dist = euclidean_distance(z1, z2)

            # ── SSIM sederhana ──
            def ssim_simple(img1, img2):
                a = img1.astype(np.float64)
                b = img2.astype(np.float64)
                mu1, mu2 = a.mean(), b.mean()
                s1, s2   = a.std(), b.std()
                cov      = np.mean((a - mu1) * (b - mu2))
                C1, C2   = (0.01 * 255)**2, (0.03 * 255)**2
                num = (2*mu1*mu2 + C1) * (2*cov + C2)
                den = (mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)
                return float(num / (den + 1e-10))

            ssim_val  = (ssim_simple(face1, face2) + 1) / 2

            # ── Tampilkan wajah ter-crop ──
            st.markdown("### 🔎 Wajah Ter-crop (Setelah Preprocessing)")
            cp1, cp2 = st.columns(2)
            cp1.image(face1, caption="Wajah 1 (100×100, CLAHE)", use_container_width=True, clamp=True)
            cp2.image(face2, caption="Wajah 2 (100×100, CLAHE)", use_container_width=True, clamp=True)

            # ── Verdict ──
            st.markdown("### 🏆 Hasil Perbandingan")
            label1 = os.path.splitext(up1.name)[0]
            label2 = os.path.splitext(up2.name)[0]

            if sim >= threshold_cos:
                verdict = "✅ MIRIP"
                verdict_class = "verdict-mirip"
            else:
                verdict = "❌ TIDAK MIRIP"
                verdict_class = "verdict-tidak"

            st.markdown(f"""
            <div class="{verdict_class}">
                <div class="verdict-text">{verdict}</div>
                <div class="verdict-sim">{sim:.4f}</div>
                <div style="color:#ccc; margin-top:6px; font-size:0.85rem;">Cosine Similarity</div>
                <div style="color:#999; margin-top:4px; font-size:0.8rem;">
                    Threshold: {threshold_cos} &nbsp;|&nbsp;
                    PC digunakan: {skip+1}–{k} &nbsp;|&nbsp;
                    Skip: {skip} PC
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📈 Metrik Detail")
            m1, m2, m3 = st.columns(3)
            m1.metric("Cosine Similarity", f"{sim:.4f}",
                      delta="≥ threshold ✅" if sim >= threshold_cos else "< threshold ❌")
            m2.metric("Euclidean Distance (PCA space)", f"{dist:.2f}")
            m3.metric("SSIM (Pixel space)", f"{ssim_val:.4f}")

            # ── Proyeksi plot ──
            st.markdown("### 📍 Proyeksi di Eigenspace")
            fig_proj = fig_projection(z1, z2, label1, label2)
            st.pyplot(fig_proj)

            # ── Info tambahan ──
            with st.expander("🔬 Detail Teknis"):
                st.markdown(f"""
                | Parameter | Nilai |
                |---|---|
                | Model dilatih dari | {st.session_state.get('model_source', '?')} |
                | Komponen PCA (k) | {pca_model.n_components_} |
                | Dimensi asli | {IMG_SIZE[0] * IMG_SIZE[1]} piksel |
                | Dimensi setelah PCA | {k - skip} |
                | Reduksi dimensi | {100*(1-(k-skip)/(IMG_SIZE[0]*IMG_SIZE[1])):.1f}% |
                | PC yang di-skip | {skip} (pencahayaan global) |
                | z₁ (5 pertama) | `{z1[:5].round(4).tolist()}` |
                | z₂ (5 pertama) | `{z2[:5].round(4).tolist()}` |
                """)

            # Cleanup temp files
            try:
                os.unlink(path1)
                os.unlink(path2)
            except Exception:
                pass


# ════════════════════════════════════════════════
#  TAB 3 — KENALI DARI DATABASE
# ════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">RECOGNIZE</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Kenali Wajah dari Database 🌐</div>',
                unsafe_allow_html=True)

    if "pca_model" not in st.session_state:
        st.warning("⚠️ Latih model terlebih dahulu di tab **Latih Model (Database)**.")
        st.stop()

    pca_model = st.session_state["pca_model"]
    mean_face = st.session_state["mean_face"]

    st.info(
        "Upload foto database (tiap foto diberi nama/label), lalu upload 1 foto query. "
        "Sistem akan mencari wajah **paling mirip** dari database."
    )

    # Upload database berlabel
    st.markdown("**📂 Upload Foto Database (beri nama file = nama orang)**")
    db_files = st.file_uploader(
        "Upload foto database (nama file = label)",
        type=["jpg","jpeg","png"],
        accept_multiple_files=True,
        key="recog_db"
    )

    st.markdown("**🔍 Upload Foto Query (yang ingin dikenali)**")
    query_file = st.file_uploader("Upload foto query", type=["jpg","jpeg","png"], key="recog_query")

    if db_files and query_file:
        if st.button("🌐 Kenali Wajah!", use_container_width=True):
            with st.spinner("Memproses..."):
                # Proses database
                db_entries = []
                for uf in db_files:
                    path_tmp = save_uploaded_to_temp(uf)
                    vec, face, pil, ok, msg = detect_face_preprocess(path_tmp)
                    label = os.path.splitext(uf.name)[0]
                    if ok:
                        z_full = pca_model.transform((vec - mean_face).reshape(1, -1))[0]
                        skip = min(pc_skip, len(z_full) - 2)
                        z = z_full[skip:]
                        db_entries.append({"label": label, "z": z, "face": face, "pil": pil})
                    try: os.unlink(path_tmp)
                    except: pass

                # Proses query
                path_q = save_uploaded_to_temp(query_file)
                vec_q, face_q, pil_q, ok_q, msg_q = detect_face_preprocess(path_q)
                try: os.unlink(path_q)
                except: pass

            if not ok_q:
                st.error(f"❌ Wajah tidak terdeteksi di foto query: {msg_q}")
                st.stop()
            if not db_entries:
                st.error("❌ Tidak ada wajah yang terdeteksi di database.")
                st.stop()

            z_q_full = pca_model.transform((vec_q - mean_face).reshape(1, -1))[0]
            skip = min(pc_skip, len(z_q_full) - 2)
            z_q = z_q_full[skip:]

            # Hitung similarity ke semua entry database
            results = []
            for entry in db_entries:
                sim = cosine_similarity(z_q, entry["z"])
                results.append((sim, entry))
            results.sort(key=lambda x: x[0], reverse=True)

            # Tampilkan query
            st.markdown("### 🔍 Foto Query")
            qc1, qc2 = st.columns([1, 3])
            qc1.image(pil_q, caption=f"Query: {query_file.name}", use_container_width=True)
            best_sim, best_entry = results[0]
            if best_sim >= threshold_cos:
                qc2.success(f"✅ Dikenali sebagai: **{best_entry['label']}** (similarity: {best_sim:.4f})")
            else:
                qc2.error(f"❌ Tidak ada wajah yang cukup mirip. Skor tertinggi: {best_sim:.4f} ({best_entry['label']})")

            # Tampilkan ranking
            st.markdown("### 📊 Ranking Kemiripan")
            for rank, (sim_val, entry) in enumerate(results[:5], 1):
                col_r1, col_r2, col_r3 = st.columns([1, 3, 1])
                col_r1.image(entry["face"], caption=entry["label"][:15], use_container_width=True, clamp=True)
                col_r2.markdown(f"**#{rank} — {entry['label']}**")
                col_r2.progress(float(max(0, sim_val)))
                verdict_tag = "✅ MIRIP" if sim_val >= threshold_cos else "❌ Tidak mirip"
                col_r3.metric("Similarity", f"{sim_val:.4f}", delta=verdict_tag)
    else:
        if not db_files:
            st.warning("👆 Upload foto database terlebih dahulu.")
        if not query_file:
            st.warning("👆 Upload foto query yang ingin dikenali.")
