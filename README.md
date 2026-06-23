# 🧠 Deteksi Kemiripan Wajah — PCA / SVD
 
> Implementasi **Principal Component Analysis (PCA) / Singular Value Decomposition (SVD)**
> berbasis **Eigenfaces** untuk membandingkan kemiripan dua wajah secara matematis.
 
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://deteksi-kemiripan-wajah-r2.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
 
---
 
## 📌 Deskripsi
 
Aplikasi ini membuktikan bahwa **wajah manusia dapat direpresentasikan sebagai vektor di ruang matematika** (eigenspace). Dua wajah yang mirip akan memiliki vektor yang berdekatan di ruang tersebut.
 
Setiap gambar wajah berukuran 100×100 piksel (= **10.000 dimensi**) direduksi menjadi **k dimensi PCA**, lalu kemiripannya dihitung menggunakan **Cosine Similarity** dan **Euclidean Distance**.
 
---
 
## ✨ Fitur
 
| Fitur | Keterangan |
|---|---|
| 🏋️ Latih Model | Bangun eigenspace dari foto database sendiri atau dataset LFW |
| 🔍 Bandingkan Dua Wajah | Upload 2 foto → hasil similarity langsung |
| 🌐 Kenali dari Database | Upload foto query → ranking mirip dari database |
| 📊 Visualisasi | Eigenfaces, singular values, cumulative variance, proyeksi eigenspace |
| ⚙️ Parameter Interaktif | Atur k, threshold, dan PC skip lewat sidebar |
| 🎨 Dark Theme | Tampilan modern, nyaman di mata |
 
---
 
## 🗂️ Struktur Proyek
 
```
deteksi-kemiripan-wajah/
│
├── app.py                  # Aplikasi Streamlit utama
├── requirements.txt        # Daftar dependensi Python
├── README.md               # Dokumentasi ini
│
├── input_images/           # (opsional) folder gambar lokal
│   ├── kecil.jpg
│   └── dewasa.jpg
│
└── output_results/         # (opsional) hasil visualisasi lokal
```
 
---
 
## ⚙️ Instalasi & Menjalankan Lokal
 
### 1. Clone repository
 
```bash
git clone https://github.com/username/deteksi-kemiripan-wajah.git
cd deteksi-kemiripan-wajah
```
 
### 2. Install dependensi
 
```bash
pip install -r requirements.txt
```
 
> **Catatan:** Gunakan `opencv-python-headless` (bukan `opencv-python`) untuk environment tanpa GUI (server / cloud).
 
### 3. Jalankan aplikasi
 
```bash
streamlit run app.py
```
 
Buka browser di `http://localhost:8501`
 
---
 
## 🚀 Deploy ke Streamlit Cloud (Gratis)
 
1. Push seluruh file ke repository **GitHub** (public atau private)
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Klik **New app** → pilih repo → pilih branch → set `Main file path: app.py`
4. Klik **Deploy** — selesai!
---
 
## 🧮 Alur Matematika
 
```
Input Gambar (100×100 px)
        │
        ▼
[1] Preprocessing
    grayscale → resize → CLAHE → normalize → flatten
    Hasil: vektor x ∈ ℝ^10000
 
        │
        ▼
[2] Centering
    Xc = X − X̄  (kurangi mean face)
 
        │
        ▼
[3] SVD / PCA
    Xc = U · Σ · Vᵀ
    Ambil k kolom pertama Vk (eigenfaces)
 
        │
        ▼
[4] Proyeksi ke Eigenspace
    z = Xc · Vk   →   z ∈ ℝ^k  (k << 10000)
 
        │
        ▼
[5] Hitung Kemiripan
    Cosine Similarity = (z₁ · z₂) / (‖z₁‖ × ‖z₂‖)
    Euclidean Distance = ‖z₁ − z₂‖
 
        │
        ▼
[6] Keputusan
    sim ≥ threshold  →  MIRIP ✅
    sim <  threshold →  TIDAK MIRIP ❌
```
 
---
 
## 🎛️ Parameter
 
| Parameter | Default | Keterangan |
|---|---|---|
| **k** (jumlah komponen PCA) | 50 | Dimensi eigenspace. Lebih besar = lebih akurat, lebih lambat |
| **Threshold Cosine Similarity** | 0.60 | Batas minimal untuk dinyatakan MIRIP |
| **PC Skip** | 3 | Jumlah PC pertama yang dilewati — PC awal menangkap pencahayaan, bukan identitas |
 
---
 
## 📦 Dependensi
 
```
streamlit>=1.32.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
matplotlib>=3.7.0
Pillow>=10.0.0
scikit-learn>=1.3.0
```
 
---
 
## 📁 Sumber Data
 
- **Dataset LFW (Labeled Faces in the Wild)** — diunduh otomatis via `sklearn.datasets.fetch_lfw_people`
  (~200MB, hanya diunduh sekali, lalu ter-cache)
- **Foto custom** — upload langsung lewat antarmuka aplikasi
---
 
## 🧠 Konsep Kunci
 
**Eigenfaces** adalah basis vektor dari ruang wajah yang diperoleh dari SVD matriks data wajah. Setiap wajah dapat direpresentasikan sebagai kombinasi linear dari eigenfaces ini.
 
**Cosine Similarity** mengukur sudut antara dua vektor di ruang eigenspace — nilainya antara -1 hingga 1, di mana 1 berarti identik dan 0 berarti tidak berkaitan sama sekali.
 
**PC Skip** — komponen utama (PC) pertama cenderung menangkap variasi pencahayaan global, bukan fitur identitas wajah. Dengan melewati beberapa PC pertama, kemiripan yang diukur lebih fokus pada struktur wajah.
 
---
 
## 👥 Anggota Kelompok
 
| No | Nama | NIM |
|---|---|---|
| 1 | ... | ... |
| 2 | ... | ... |
| 3 | ... | ... |
 
**Kelompok 3 — Teknik Informatika, Universitas Negeri Semarang**
Mata Kuliah: Aljabar Linear dan Matriks
 
---
 
## 📄 Lisensi
 
MIT License — bebas digunakan dan dimodifikasi dengan menyertakan atribusi.
 
